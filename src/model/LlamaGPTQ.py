import json
import logging
import os
from dataclasses import dataclass, field, fields
from os.path import isfile, join
from typing import Dict, Optional, Union

import accelerate
import torch
import torch.nn as nn
import transformers

from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel
from transformers.modeling_utils import no_init_weights
from transformers.utils.generic import ContextManagers
from transformers.utils.hub import (
    PushToHubMixin,
    cached_file
)

from src.model.utils import find_layers, simple_dispatch_model, make_quant
from src.model.fused_llama_attn import FusedLlamaAttentionForQuantizedModel
from src.model.fused_llama_mlp import FusedLlamaMLPForQuantizedModel

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

SYNONYMS = {
    "w_bit": "bits",
    "q_group_size": "group_size",
}

@dataclass
class BaseQuantizeConfig(PushToHubMixin):
    bits: int = field(default=4, metadata={"choices": [2, 3, 4, 8]})
    group_size: int = field(default=-1)
    damp_percent: float = field(default=0.01)
    desc_act: bool = field(default=True)
    static_groups: bool = field(default=False)
    sym: bool = field(default=True)
    true_sequential: bool = field(default=True)
    is_marlin_format: bool = field(default=False)
    model_name_or_path: Optional[str] = field(default=None)
    model_file_base_name: Optional[str] = field(default=None)
    awq_gemm_checkpoint: Optional[bool] = field(default=False)

    def __post_init__(self):
        fields_info = fields(self)

        if self.bits not in fields_info[0].metadata["choices"]:
            raise ValueError(f"only support quantize to {fields_info[0].metadata['choices']} bits.")
        if self.group_size != -1 and self.group_size <= 0:
            raise ValueError("unless equal to -1, group_size must greater then 0.")
        if not (0 < self.damp_percent < 1):
            raise ValueError("damp_percent must between 0 and 1.")

    def save_pretrained(self, save_dir: str, **kwargs):
        with open(join(save_dir, "quantize_config.json"), "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_pretrained(cls, save_dir: str, **kwargs):
        # Parameters related to loading from Hugging Face Hub
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", False)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", None)
        commit_hash = kwargs.pop("_commit_hash", None)

        transformers_config = False
        for quantize_config_filename in [
            "quantize_config.json",
            "quant_config.json",
            "config.json",
        ]:
            if os.path.isdir(save_dir):  # Local
                resolved_config_file = join(save_dir, quantize_config_filename)
            else:  # Remote
                resolved_config_file = cached_file(
                    save_dir,
                    quantize_config_filename,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    use_auth_token=use_auth_token,
                    revision=revision,
                    local_files_only=local_files_only,
                    subfolder=subfolder,
                    _raise_exceptions_for_missing_entries=False,
                    _raise_exceptions_for_connection_errors=False,
                    _commit_hash=commit_hash,
                )
            if resolved_config_file is not None:
                if quantize_config_filename == "config.json":
                    transformers_config = True
                break

        if resolved_config_file is None:
            raise ValueError(
                "No quantize_config.json, quant_config.json or config.json file was found in the model repository."
            )

        field_names = [field.name for field in fields(cls)]
        with open(resolved_config_file, "r", encoding="utf-8") as f:
            args_from_json = json.load(f)

            if transformers_config:
                args_from_json = args_from_json["quantization_config"]

            filtered_args = {"awq_gemm_checkpoint": False}
            for key, val in args_from_json.items():
                if key == "version" and val == "GEMM":
                    filtered_args["awq_gemm_checkpoint"] = True
                elif key in field_names:
                    filtered_args[key] = val
                elif key in SYNONYMS and SYNONYMS[key] in field_names:
                    filtered_args[SYNONYMS[key]] = val
                else:
                    logger.warning(f"ignoring unknown parameter in {quantize_config_filename}: {key}.")

            if filtered_args["awq_gemm_checkpoint"]:
                # AWQ does not reorder the rows.
                filtered_args["desc_act"] = False

            if "sym" not in args_from_json:
                logger.warning(
                    f"The quantization configuration {quantize_config_filename} does not contain an entry `sym` (symetric quantization). This may result in silent errors."
                )

            return cls(**filtered_args)

    def to_dict(self):
        return {
            "bits": self.bits,
            "group_size": self.group_size,
            "damp_percent": self.damp_percent,
            "desc_act": self.desc_act,
            "static_groups": self.static_groups,
            "sym": self.sym,
            "true_sequential": self.true_sequential,
            "model_name_or_path": self.model_name_or_path,
            "model_file_base_name": self.model_file_base_name,
            "is_marlin_format": self.is_marlin_format,
            "quant_method": "gptq",
        }

class LlamaGPTQ(nn.Module, PushToHubMixin):
    layer_type = "LlamaDecoderLayer"
    layers_block_name = "model.layers"
    outside_layer_modules = ["model.embed_tokens", "model.norm"]
    inside_layer_modules = [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.o_proj"],
        ["mlp.up_proj", "mlp.gate_proj"],
        ["mlp.down_proj"],
    ]

    fused_attn_module_type = FusedLlamaAttentionForQuantizedModel
    fused_mlp_module_type = FusedLlamaMLPForQuantizedModel
    lm_head_name: str = "lm_head"

    def __init__(
        self,
        model: PreTrainedModel,
        quantized: bool,
        quantize_config: BaseQuantizeConfig,
        is_triton_backend: bool = False,
        injected_fused_attention: bool = False,
        injected_fused_mlp: bool = False,
    ):
        super().__init__()

        self.model = model
        self.model_type = self.model.config.model_type
        self._quantized = quantized
        self.quantize_config = quantize_config
        self.config = self.model.config

        self.is_triton_backend = is_triton_backend
        self.injected_fused_attention = injected_fused_attention
        self.injected_fused_mlp = injected_fused_mlp
        
    @property
    def quantized(self):
        return self._quantized

    @property
    def hf_device_map(self):
        return getattr(self.model, "hf_device_map", None)

    @property
    def device(self):
        if not self.hf_device_map:
            return self.model.device
        else:
            device = [d for d in self.hf_device_map.values() if d not in {"disk"}][0]
            return torch.device(device)

    def to(self, device: Union[str, torch.device]):
        self.model.to(device)
        return self

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def generate(self, **kwargs):
        """shortcut for model.generate"""
        with torch.inference_mode(), torch.amp.autocast(device_type=self.device.type):
            return self.model.generate(**kwargs)

    @classmethod
    def from_quantized(
        cls,
        model_name_or_path: Optional[str],
        device_map: Optional[Union[str, Dict[str, Union[int, str]]]] = None,
        max_memory: Optional[dict] = None,
        device: Optional[Union[str, int]] = None,
        use_triton: bool = True,
        torch_dtype: Optional[torch.dtype] = None,
        inject_fused_attention: bool = True,
        inject_fused_mlp: bool = True,
        use_cuda_fp16: bool = True,
        quantize_config: Optional[BaseQuantizeConfig] = None,
        model_basename: Optional[str] = None,
        trust_remote_code: bool = False,
        warmup_triton: bool = False,
        **kwargs,
    ):
        # == step1: prepare configs and file names == #
        config = AutoConfig.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code
        )

        if config.model_type not in ['llama']:
            raise TypeError(f"{config.model_type} isn't supported yet.")

        if quantize_config is None:
            quantize_config = BaseQuantizeConfig.from_pretrained(model_name_or_path, **kwargs)

        if model_basename is None:
            if quantize_config.model_file_base_name:
                possible_model_basenames = [quantize_config.model_file_base_name]
            else:
                possible_model_basenames = [
                    f"gptq_model-{quantize_config.bits}bit-{quantize_config.group_size}g",
                    "model",
                ]
        else:
            possible_model_basenames = [model_basename]

        quantize_config.model_name_or_path = model_name_or_path

        extensions = [".safetensors",]
        model_name_or_path = str(model_name_or_path)

        resolved_archive_file = None
        true_model_basename = None
        searched_files = []

        for ext in extensions:
            for possible_model_basename in possible_model_basenames:
                model_save_name = join(model_name_or_path, possible_model_basename)
                searched_files.append(possible_model_basename + ext)
                if isfile(model_save_name + ext):
                    resolved_archive_file = model_save_name + ext
                    true_model_basename = possible_model_basename
                    break

        quantize_config.model_file_base_name = true_model_basename
        if resolved_archive_file is None:
            raise FileNotFoundError(
                f"Could not find a model in {model_name_or_path} with a name in {', '.join(searched_files)}. Please specify the argument model_basename to use a custom file name."
            )

        model_save_name = resolved_archive_file

        # == step2: convert model to gptq-model (replace Linear with QuantLinear) == #
        def skip(*args, **kwargs):
            pass

        torch_dtype = torch.float16

        torch.nn.init.kaiming_uniform_ = skip
        torch.nn.init.uniform_ = skip
        torch.nn.init.normal_ = skip

        transformers.modeling_utils._init_weights = False

        init_contexts = [no_init_weights()]
        with ContextManagers(init_contexts):
            model = AutoModelForCausalLM.from_config(
                config, trust_remote_code=trust_remote_code, torch_dtype=torch_dtype
            )

            layers = find_layers(model)
            ignore_layers = [cls.lm_head_name] + cls.outside_layer_modules
            for name in list(layers.keys()):
                if any(name.startswith(ignore_layer) for ignore_layer in ignore_layers) or all(
                    not name.endswith(ignore_layer)
                    for sublist in cls.inside_layer_modules
                    for ignore_layer in sublist
                ):
                    logger.info(f"The layer {name} is not quantized.")
                    del layers[name]

            make_quant(
                model,
                layers,
                quantize_config.bits,
                quantize_config.group_size,
                use_triton=use_triton,
                use_cuda_fp16=use_cuda_fp16,
                desc_act=quantize_config.desc_act,
            )
            model.tie_weights()

        # == step3: load checkpoint and dispatch == #
        if isinstance(device_map, str) and device_map not in [
            "auto",
            "balanced",
            "balanced_low_0",
            "sequential",
        ]:
            raise ValueError(
                "If passing a string for `device_map`, please choose 'auto', 'balanced', 'balanced_low_0' or "
                "'sequential'."
            )

        device = torch.device(device)
        if not max_memory and not device_map:
            device_map = {"": device.index if device.type == "cuda" else device.type}


        accelerate.utils.modeling.load_checkpoint_in_model(
            model,
            dtype=torch_dtype,
            checkpoint=model_save_name,
            device_map=device_map,
            offload_state_dict=True,
            offload_buffers=True,
        )

        # TODO: Why are we using this custom function and not dispatch_model?
        model = simple_dispatch_model(model, device_map)
    
        # == step4: set seqlen == #
        model_config = model.config.to_dict()
        seq_len_keys = ["max_position_embeddings", "seq_length", "n_positions"]
        if any(k in model_config for k in seq_len_keys):
            for key in seq_len_keys:
                if key in model_config:
                    model.seqlen = model_config[key]
                    break
        else:
            logger.warning("can't get model's sequence length from model config, will set to 4096.")
            model.seqlen = 4096

        # == step5: (optional) inject optimized module == #
        if inject_fused_attention:
            if cls.fused_attn_module_type is None:
                inject_fused_attention = False
                logger.warning(f"{cls.__name__} hasn't fused attention module yet, will skip inject fused attention.")
            else:
                cls.fused_attn_module_type.inject_to_model(
                    model,
                    use_triton=use_triton,
                    group_size=quantize_config.group_size,
                    use_cuda_fp16=use_cuda_fp16,
                    desc_act=quantize_config.desc_act,
                    trainable=False,
                    bits=quantize_config.bits,
                    disable_exllama=True,
                    disable_exllamav2=True,
                )
        if inject_fused_mlp:
            if cls.fused_mlp_module_type is None:
                inject_fused_mlp = False
                logger.warning(f"{cls.__name__} hasn't fused mlp module yet, will skip inject fused mlp.")
            else:
                cls.fused_mlp_module_type.inject_to_model(model, use_triton=use_triton)

        torch.cuda.empty_cache()
        model.eval()

        # == step6: (optional) warmup triton == #
        if use_triton and warmup_triton:
            from src.model.QuantLinear import QuantLinear
            QuantLinear.warmup(model, seqlen=model.seqlen)

            if inject_fused_mlp and cls.fused_mlp_module_type is not None:
                cls.fused_mlp_module_type.warmup(model, seqlen=model.seqlen)


        return cls(
            model,
            True,
            quantize_config,
            is_triton_backend=use_triton,
            injected_fused_attention=inject_fused_attention,
            injected_fused_mlp=inject_fused_mlp and use_triton,
        )

    # def warmup_triton(self, enabled: bool = True):
    #     if not enabled:
    #         return
    #     if not TRITON_AVAILABLE:
    #         logger.warning("triton is not available, skip warmup stage directly.")
    #         return

    #     from ..nn_modules.qlinear.qlinear_triton import QuantLinear

    #     QuantLinear.warmup(self.model, seqlen=self.model.seqlen)

    #     if self.fused_mlp_module_type is not None:
    #         self.fused_mlp_module_type.warmup(self.model, seqlen=self.model.seqlen)

    # def enable_trainable_mode(self, enabled: bool = True):
    #     if not self.is_triton_backend and enabled:
    #         raise NotImplementedError("For now, trainable mode only supports triton backend.")
    #     for n, m in self.model.named_modules():
    #         if hasattr(m, "trainable"):
    #             setattr(m, "trainable", enabled)

    # def disable_trainable_mode(self):
    #     self.enable_trainable_mode(enabled=False)


    # def __getattr__(self, item):
    #     try:
    #         return super().__getattr__(item)
    #     except Exception:
    #         return getattr(self.model, item)


__all__ = ["LlamaGPTQ",]
