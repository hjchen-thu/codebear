import copy
import json
import logging
import os
from dataclasses import dataclass, field, fields
from os.path import isdir, isfile, join
from typing import Dict, List, Optional, Union

import accelerate
import huggingface_hub
import torch
import torch.nn as nn
import transformers
from accelerate.hooks import remove_hook_from_module
from safetensors import safe_open
from safetensors.torch import load_file as safe_load
from safetensors.torch import save_file as safe_save
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel
from transformers.modeling_utils import no_init_weights
from transformers.utils.generic import ContextManagers
from transformers.utils.hub import (
    CommitOperationAdd,
    PushToHubMixin,
    cached_file,
    create_commit,
    create_repo,
)


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


def find_layers(module, layers=None, name=""):
    if not layers:
        layers = [transformers.pytorch_utils.Conv1D, nn.Conv2d, nn.Linear]
    for layer in layers:
        if isinstance(module, layer):
            return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(child, layers=layers, name=name + "." + name1 if name != "" else name1))
    return res


def get_device(obj: Union[torch.Tensor, nn.Module]):
    if isinstance(obj, torch.Tensor):
        return obj.device
    return next(obj.parameters()).device


def dynamically_import_QuantLinear(
    use_triton: bool,
    desc_act: bool,
    group_size: int,
    bits: int,
    use_qigen: bool = False,
    disable_marlin: bool = True,
):

    from auto_gptq.nn_modules.qlinear.qlinear_triton import QuantLinear    
    return QuantLinear


def get_module_by_name_suffix(model, module_name: str):
    for name, module in model.named_modules():
        if name.endswith(module_name):
            return module


def simple_dispatch_model(model, device_map):
    from accelerate.hooks import AlignDevicesHook, add_hook_to_module

    if "" in device_map:
        d = device_map[""]
        model = model.to(torch.device(d))
        model.hf_device_map = device_map
        return model

    tied_params = accelerate.utils.modeling.find_tied_parameters(model)
    if set(device_map.values()) == {"cpu"} or set(device_map.values()) == {
        "cpu",
        "disk",
    }:
        main_device = "cpu"
    else:
        main_device = [d for d in device_map.values() if d not in ["cpu", "disk"]][0]

    cpu_offload_group = [(n, d) for n, d in device_map.items() if d == "cpu"]
    prev_hook = None
    for idx, (n, d) in enumerate(cpu_offload_group):
        m = get_module_by_name_suffix(model, n)
        _, prev_hook = accelerate.cpu_offload_with_hook(m, execution_device=main_device, prev_module_hook=prev_hook)
    # set first cpu offload module's prev_module_hook to the last cpu offload module's hook
    if len(cpu_offload_group) > 1:
        get_module_by_name_suffix(model, cpu_offload_group[0][0])._hf_hook.prev_module_hook = prev_hook

    for n, d in device_map.items():
        m = get_module_by_name_suffix(model, n)
        if d != "cpu":
            d = torch.device(d)
            hook = AlignDevicesHook(d, io_same_device=True, place_submodules=True)
            add_hook_to_module(m, hook)
    accelerate.utils.modeling.retie_parameters(model, tied_params)
    model.hf_device_map = device_map

    return model




def autogptq_post_init(model, use_act_order: bool, max_input_length: Optional[int] = None):

    torch.cuda.empty_cache()

    return model





def make_quant(
    module,
    names,
    bits,
    group_size,
    name="",
    use_triton: bool = False,
    use_marlin: bool = False,
    use_qigen: bool = False,
    use_cuda_fp16: bool = True,
    desc_act: bool = False,
    trainable: bool = False,
):

    QuantLinear = dynamically_import_QuantLinear(
        use_triton=use_triton,
        desc_act=desc_act,
        group_size=group_size,
        bits=bits,
        disable_marlin=not use_marlin,
        use_qigen=use_qigen,
    )

    if isinstance(module, QuantLinear):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        name1 = name + "." + attr if name != "" else attr
        if name1 in names:
            ori_layer_device = get_device(getattr(module, attr))
            delattr(module, attr)
            if isinstance(tmp, nn.Linear):
                in_features = tmp.in_features
                out_features = tmp.out_features
            elif isinstance(tmp, nn.Conv2d):
                in_features = tmp.in_channels
                out_features = tmp.out_channels
            elif isinstance(tmp, transformers.pytorch_utils.Conv1D):
                in_features = tmp.weight.shape[0]
                out_features = tmp.weight.shape[1]
            if (not (desc_act) or group_size == -1) and not use_triton and not use_qigen:
                new_layer = QuantLinear(
                    bits,
                    group_size,
                    in_features,
                    out_features,
                    True,
                    use_cuda_fp16=use_cuda_fp16,
                    trainable=trainable,
                    weight_dtype=tmp.weight.dtype,
                )
            else:
                new_layer = QuantLinear(
                    bits,
                    group_size,
                    in_features,
                    out_features,
                    True,
                    trainable=trainable,
                    weight_dtype=tmp.weight.dtype,
                )
            new_layer.device = ori_layer_device
            setattr(module, attr, new_layer.to(ori_layer_device))
    for name1, child in module.named_children():
        make_quant(
            child,
            names,
            bits,
            group_size,
            name + "." + name1 if name != "" else name1,
            use_triton=use_triton,
            use_cuda_fp16=use_cuda_fp16,
            desc_act=desc_act,
            trainable=trainable,
            use_qigen=use_qigen,
        )




class GeneralQuantLinear(nn.Linear):
    def __init__(self, quant_linear_module):
        super().__init__(
            in_features=quant_linear_module.infeatures,
            out_features=quant_linear_module.outfeatures,
            bias=True,
        )
        self.infeatures = quant_linear_module.infeatures
        self.outfeatures = quant_linear_module.outfeatures
        self.bits = quant_linear_module.bits
        self.group_size = quant_linear_module.group_size
        self.maxq = quant_linear_module.maxq

        self.weight.requires_grad = False

        self.weight.data = quant_linear_module.qweight
        self.register_buffer("qweight", quant_linear_module.qweight)
        self.bias.data = quant_linear_module.bias

        self.qweight.requires_grad = False
        self.bias.requires_grad = False

        self.register_buffer("qzeros", quant_linear_module.qzeros)
        self.register_buffer("scales", quant_linear_module.scales)
        self.register_buffer("g_idx", quant_linear_module.g_idx)

        if hasattr(quant_linear_module, "wf"):
            self.wf = quant_linear_module.wf
        if hasattr(quant_linear_module, "kernel_switch_threshold"):
            self.kernel_switch_threshold = quant_linear_module.kernel_switch_threshold
        if hasattr(quant_linear_module, "autogptq_cuda_available"):
            self.autogptq_cuda_available = quant_linear_module.autogptq_cuda_available

        self.trainable = quant_linear_module.trainable

        self.forward = quant_linear_module.forward

    @classmethod
    def inject_to_model(cls, model, target_module_type):
        for name, m in model.named_modules():
            if not isinstance(m, target_module_type):
                continue
            new_m = cls(m)
            if "." in name:
                parent_name = name.rsplit(".", 1)[0]
                child_name = name[len(parent_name) + 1 :]
                parent = model.get_submodule(parent_name)
            else:
                parent_name = ""
                parent = model
                child_name = name

            setattr(parent, child_name, new_m)




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

    # fused_attn_module_type = FusedLlamaAttentionForQuantizedodel
    # fused_mlp_module_type = FusedLlamaMLPForQuantizedModelM
    lm_head_name: str = "lm_head"

    def __init__(
        self,
        model: PreTrainedModel,
        quantized: bool,
        quantize_config: BaseQuantizeConfig,
        is_triton_backend: bool = False,
        injected_fused_attention: bool = False,
        injected_fused_mlp: bool = False,
        trainable: bool = False,
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
        self.trainable = trainable

    @property
    def quantized(self):
        return self._quantized

    @property
    def hf_device_map(self):
        return getattr(self.model, "hf_device_map", None)

    # def _prepare_examples_for_quantization(
    #     self,
    #     examples: List[Dict[str, Union[List[int], torch.LongTensor]]],
    #     batch_size: int = 1,
    # ):
    #     def _convert_tensor_to_list(tensor):
    #         if isinstance(tensor, torch.Tensor):
    #             if len(tensor.shape) == 1:
    #                 tensor = tensor.unsqueeze(0)
    #             tensor = tensor.long()
    #             return tensor.cpu().numpy().tolist()
    #         return [tensor]

    #     new_examples = []
    #     for example in examples:
    #         input_ids = _convert_tensor_to_list(example["input_ids"])
    #         attention_mask = _convert_tensor_to_list(example["attention_mask"])
    #         if "labels" in example:
    #             labels = _convert_tensor_to_list(example["labels"])
    #         elif "label" in example:
    #             labels = _convert_tensor_to_list(example["label"])
    #         elif "label_ids" in example:
    #             labels = _convert_tensor_to_list(example["label_ids"])
    #         else:
    #             labels = copy.deepcopy(input_ids)
    #         new_examples.append(
    #             {
    #                 "input_ids": input_ids,
    #                 "attention_mask": attention_mask,
    #                 "labels": labels,
    #             }
    #         )
    #     pad_token_id = self.config.pad_token_id
    #     if not pad_token_id:
    #         pad_token_id = self.config.eos_token_id

    #     new_examples = [
    #         collate_data(new_examples[start : start + batch_size], pad_token_id)
    #         for start in range(0, len(new_examples), batch_size)
    #     ]
    #     for new_example in new_examples:
    #         del new_example["labels"]

    #     return new_examples

    # @torch.inference_mode()
    # def quantize(
    #     self,
    #     examples: List[Dict[str, Union[List[int], torch.LongTensor]]],
    #     batch_size: int = 1,
    #     use_triton: bool = False,
    #     use_cuda_fp16: bool = True,
    #     autotune_warmup_after_quantized: bool = False,
    #     cache_examples_on_gpu: bool = True,
    # ):
    #     if self.quantized:
    #         raise EnvironmentError("can't execute quantize because the model is quantized.")
    #     if use_triton and not TRITON_AVAILABLE:
    #         logger.warning("triton is not installed, reset use_triton to False")
    #         use_triton = False

    #     device_map = self.hf_device_map
    #     if device_map:
    #         for name, device in device_map.items():
    #             if device == "cpu":
    #                 logger.info(f"truly offloading {name} to cpu with hook.")
    #                 module = get_module_by_name_suffix(self.model, name)
    #                 remove_hook_from_module(module, recurse=True)
    #                 accelerate.cpu_offload_with_hook(module, CUDA_0)

    #     layer_inputs = []
    #     attention_masks = []
    #     position_ids = []
    #     layer_input_kwargs = []
    #     layer_outputs = []

    #     examples = self._prepare_examples_for_quantization(examples, batch_size)

    #     def nested_move_to_device(v, device):
    #         if isinstance(v, torch.Tensor):
    #             return move_to_device(v, device)
    #         elif isinstance(v, (list, tuple)):
    #             return type(v)([nested_move_to_device(e, device) for e in v])
    #         else:
    #             return v

    #     class LayerHijacker(nn.Module):
    #         """hijack layer's forward pass to cache data"""

    #         def __init__(self, m, device):
    #             super().__init__()
    #             self.module = m
    #             self.data_device = device if cache_examples_on_gpu else CPU

    #         def forward(self, inp=None, **kwargs):
    #             if inp is None:  # some models use all key-value arguments in forward pass call
    #                 for kwarg_name in ["hidden_states"]:
    #                     if kwarg_name in kwargs:
    #                         inp = kwargs[kwarg_name]
    #                         break
    #             layer_inputs.append(move_to_device(inp, self.data_device))

    #             if kwargs["attention_mask"] is not None:
    #                 attention_masks.append(kwargs["attention_mask"].to(self.data_device))
    #             else:
    #                 attention_masks.append(None)

    #             pos_ids = kwargs.get("position_ids", None)
    #             if pos_ids is not None:
    #                 position_ids.append(move_to_device(pos_ids, self.data_device))
    #             one_kwargs = {}
    #             for (
    #                 k,
    #                 v,
    #             ) in kwargs.items():  # make sure other arguments also be captured
    #                 if k not in ["hidden_states", "attention_mask", "position_ids"]:
    #                     one_kwargs[k] = nested_move_to_device(v, self.data_device)
    #             layer_input_kwargs.append(one_kwargs)
    #             raise ValueError

    #     forward_pass_use_cache = self.model.config.use_cache
    #     self.model.config.use_cache = False

    #     num_batches = len(examples)
    #     layers = get_module_by_name_prefix(self.model, self.layers_block_name)

    #     force_layer_back_to_cpu = False
    #     if get_device(layers[0]) == CPU:
    #         layers[0] = layers[0].to(CUDA_0)
    #         force_layer_back_to_cpu = True

    #     cur_layer_device = get_device(layers[0])
    #     ori_outside_layer_module_devices = {}
    #     for module_name in self.outside_layer_modules:
    #         module = get_module_by_name_prefix(self.model, module_name)

    #         if module is None:
    #             continue

    #         ori_outside_layer_module_devices[module_name] = get_device(module)
    #         if module is not None:
    #             move_to_device(module, cur_layer_device)

    #     # get inputs for first layer
    #     layers[0] = LayerHijacker(layers[0], cur_layer_device)
    #     for example in examples:
    #         for k, v in example.items():
    #             if len(v.shape) == 1:
    #                 v = v.unsqueeze(0)
    #             example[k] = move_to_device(v, cur_layer_device)
    #         try:
    #             self.model(**example)
    #         except ValueError:
    #             pass
    #     layers[0] = layers[0].module

    #     move_to_device(layers[0], CPU if force_layer_back_to_cpu else cur_layer_device)
    #     for module_name in self.outside_layer_modules:
    #         module = get_module_by_name_prefix(self.model, module_name)
    #         if module is not None:
    #             move_to_device(module, ori_outside_layer_module_devices[module_name])

    #     torch.cuda.empty_cache()

    #     inside_layer_modules = self.inside_layer_modules
    #     if not self.quantize_config.true_sequential:
    #         inside_layer_modules = [sum(inside_layer_modules, [])]
    #     quantizers = {}
    #     for i in range(len(layers)):
    #         logger.info(f"Start quantizing layer {i + 1}/{len(layers)}")
    #         layer = layers[i]
    #         force_layer_back_to_cpu = False
    #         if get_device(layer) == CPU:
    #             move_to_device(layer, CUDA_0)
    #             force_layer_back_to_cpu = True
    #         cur_layer_device = get_device(layer)

    #         full = find_layers(layer)
    #         for names in inside_layer_modules:
    #             subset = {n: full[n] for n in names if n in full}
    #             gptq = {}
    #             for name in subset:
    #                 gptq[name] = GPTQ(subset[name])
    #                 gptq[name].quantizer.configure(
    #                     self.quantize_config.bits,
    #                     perchannel=True,
    #                     sym=self.quantize_config.sym,
    #                     mse=False,
    #                 )

    #             def add_batch(name):
    #                 def tmp(_, inp, out):
    #                     # gptq is mutable.
    #                     gptq[name].add_batch(inp[0].data, out.data)  # noqa: F821

    #                 return tmp

    #             handles = []
    #             for name in subset:
    #                 handles.append(subset[name].register_forward_hook(add_batch(name)))
    #             for j in range(num_batches):
    #                 layer_input = move_to_device(layer_inputs[j], cur_layer_device)
    #                 layer_attention_mask = move_to_device(attention_masks[j], cur_layer_device)
    #                 additional_layer_inputs = {"attention_mask": layer_attention_mask}
    #                 layer_position_ids = (
    #                     None if not position_ids else move_to_device(position_ids[j], cur_layer_device)
    #                 )
    #                 if layer_position_ids is not None:
    #                     additional_layer_inputs["position_ids"] = layer_position_ids
    #                 for k, v in layer_input_kwargs[j].items():
    #                     additional_layer_inputs[k] = nested_move_to_device(v, cur_layer_device)
    #                 layer(layer_input, **additional_layer_inputs)
    #             for h in handles:
    #                 h.remove()

    #             for name in subset:
    #                 logger.info(f"Quantizing {name} in layer {i + 1}/{len(layers)}...")
    #                 scale, zero, g_idx = gptq[name].fasterquant(
    #                     percdamp=self.quantize_config.damp_percent,
    #                     group_size=self.quantize_config.group_size,
    #                     actorder=self.quantize_config.desc_act,
    #                     static_groups=self.quantize_config.static_groups,
    #                 )
    #                 quantizers[f"{self.layers_block_name}.{i}.{name}"] = (
    #                     gptq[name].quantizer.to(CPU if force_layer_back_to_cpu else cur_layer_device),
    #                     move_to_device(scale, CPU if force_layer_back_to_cpu else cur_layer_device),
    #                     move_to_device(zero, CPU if force_layer_back_to_cpu else cur_layer_device),
    #                     move_to_device(g_idx, CPU if force_layer_back_to_cpu else cur_layer_device),
    #                 )
    #                 gptq[name].free()

    #         for j in range(num_batches):
    #             layer_input = move_to_device(layer_inputs[j], cur_layer_device)
    #             layer_attention_mask = move_to_device(attention_masks[j], cur_layer_device)
    #             additional_layer_inputs = {"attention_mask": layer_attention_mask}
    #             layer_position_ids = None if not position_ids else move_to_device(position_ids[j], cur_layer_device)
    #             if layer_position_ids is not None:
    #                 additional_layer_inputs["position_ids"] = layer_position_ids
    #             for k, v in layer_input_kwargs[j].items():
    #                 additional_layer_inputs[k] = nested_move_to_device(v, cur_layer_device)
    #             layer_output = move_to_device(
    #                 layer(layer_input, **additional_layer_inputs)[0],
    #                 cur_layer_device if cache_examples_on_gpu else CPU,
    #             )
    #             layer_outputs.append(layer_output)

    #         layers[i] = move_to_device(layer, CPU if force_layer_back_to_cpu else cur_layer_device)
    #         del layer
    #         del gptq
    #         del layer_inputs
    #         layer_inputs, layer_outputs = layer_outputs, []
    #         torch.cuda.empty_cache()

    #     pack_model(
    #         model=self.model,
    #         quantizers=quantizers,
    #         bits=self.quantize_config.bits,
    #         group_size=self.quantize_config.group_size,
    #         use_triton=use_triton,
    #         use_cuda_fp16=use_cuda_fp16,
    #         desc_act=self.quantize_config.desc_act,
    #         warmup_triton=autotune_warmup_after_quantized,
    #         force_layer_back_to_cpu=force_layer_back_to_cpu,
    #     )
    #     if device_map:
    #         self.model = remove_hook_from_module(self.model, recurse=True)
    #         self.model = simple_dispatch_model(self.model, device_map)
    #     self.model.config.use_cache = forward_pass_use_cache

    #     self._quantized = True

    #     torch.cuda.empty_cache()


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

    # def prepare_inputs_for_generation(self, *args, **kwargs):
    #     """shortcut for model.prepare_inputs_for_generation"""
    #     return self.model.prepare_inputs_for_generation(*args, **kwargs)

    # def push_to_hub(
    #     self,
    #     repo_id: str,
    #     save_dir: Optional[str] = None,
    #     use_safetensors: Optional[bool] = True,
    #     safetensors_metadata: Optional[Dict[str, str]] = None,
    #     commit_message: Optional[str] = "Upload of AutoGPTQ quantized model",
    #     use_auth_token: Optional[Union[bool, str]] = None,
    #     private: Optional[bool] = None,
    #     token: Optional[Union[bool, str]] = None,
    #     create_pr: Optional[bool] = False,
    # ) -> str:
    #     """
    #     Upload the model to the Hugging Face Hub.

    #     Parameters:
    #         repo_id (`str`):
    #             The name of the repository you want to push your tool to. It should contain your organization name when
    #             pushing to a given organization.
    #         save_dir (`str`, *optional*):
    #             The name of the local folder to save the model to.
    #             If the model has already been saved, this parameter can be omitted.
    #         use_safetensors (`bool`, *optional*):
    #             Save the model using `safetensors`.
    #             If the model has already been saved, this parameter can be omitted.
    #         safetensors_metadata: (`dict`, *optional*, defaults to `None`):
    #             Pass optional metadata dictionary to be saved in the `safetensors` model file(s).
    #             Metadata is optional and is purely for informational purposes. It does not affect inference.
    #             If `None`, no metadata will be saved.
    #         commit_message (`str`, *optional*, defaults to `"Upload tool"`):
    #             Message to commit while pushing.
    #         use_auth_token (`bool` or `str`, *optional*):
    #             The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
    #             when running `huggingface-cli login` (stored in `~/.huggingface`). Will default to `True` if `repo_url`
    #             is not specified.
    #         private (`bool`, *optional*):
    #             Whether or not the repository created should be private.
    #         token (`bool` or `str`, *optional*):
    #             The token to use as HTTP bearer authorization for remote files. If unset, will use the token generated
    #             when running `huggingface-cli login` (stored in `~/.huggingface`).
    #         create_pr (`bool`, *optional*, defaults to `False`):
    #             Whether or not to create a PR with the uploaded files or directly commit.
    #     """
    #     if (
    #         self.quantize_config.model_name_or_path is None or not isdir(self.quantize_config.model_name_or_path)
    #     ) and save_dir is None:
    #         raise ValueError(
    #             "Quantized model should be saved first, or you can provide save_dir to make sure model is saved to local disk before uploading."
    #         )

    #     if save_dir is not None:
    #         logger.info(f"Saving model to {save_dir}")
    #         self.save_quantized(save_dir, use_safetensors, safetensors_metadata)

    #     repo_url = create_repo(
    #         repo_id=repo_id,
    #         token=token,
    #         private=private,
    #         exist_ok=True,
    #         repo_type="model",
    #     )
    #     repo_id = repo_url.repo_id

    #     if self.quantize_config.model_name_or_path is not None:
    #         work_dir = self.quantize_config.model_name_or_path
    #         operations = [
    #             CommitOperationAdd(path_or_fileobj=join(work_dir, f), path_in_repo=f) for f in os.listdir(work_dir)
    #         ]
    #         logger.info(f"Uploading the following files to {repo_id}: {','.join(os.listdir(work_dir))}")
    #         return create_commit(
    #             repo_id=repo_id,
    #             operations=operations,
    #             commit_message=commit_message,
    #             token=use_auth_token,
    #             create_pr=create_pr,
    #             repo_type="model",
    #         )

    # def save_quantized(
    #     self,
    #     save_dir: str,
    #     use_safetensors: bool = True,
    #     safetensors_metadata: Optional[Dict[str, str]] = None,
    # ):
    #     """save quantized model and configs to local disk"""
    #     os.makedirs(save_dir, exist_ok=True)

    #     if not self.quantized:
    #         raise EnvironmentError("can only save quantized model, please execute .quantize first.")

    #     self.model.to(CPU)

    #     model_base_name = (
    #         self.quantize_config.model_file_base_name
    #         or f"gptq_model-{self.quantize_config.bits}bit-{self.quantize_config.group_size}g"
    #     )
    #     if use_safetensors:
    #         model_save_name = model_base_name + ".safetensors"
    #         state_dict = self.model.state_dict()
    #         state_dict = {k: v.clone().contiguous() for k, v in state_dict.items()}
    #         if safetensors_metadata is None:
    #             safetensors_metadata = {}
    #         elif not isinstance(safetensors_metadata, dict):
    #             raise TypeError("safetensors_metadata must be a dictionary.")
    #         else:
    #             logger.debug(f"Received safetensors_metadata: {safetensors_metadata}")
    #             new_safetensors_metadata = {}
    #             converted_keys = False
    #             for key, value in safetensors_metadata.items():
    #                 if not isinstance(key, str) or not isinstance(value, str):
    #                     converted_keys = True
    #                     try:
    #                         new_key = str(key)
    #                         new_value = str(value)
    #                     except Exception as e:
    #                         raise TypeError(
    #                             f"safetensors_metadata: both keys and values must be strings and an error occured when trying to convert them: {e}"
    #                         )
    #                     if new_key in new_safetensors_metadata:
    #                         logger.warning(
    #                             f"After converting safetensors_metadata keys to strings, the key '{new_key}' is duplicated. Ensure that all your metadata keys are strings to avoid overwriting."
    #                         )
    #                     new_safetensors_metadata[new_key] = new_value
    #             safetensors_metadata = new_safetensors_metadata
    #             if converted_keys:
    #                 logger.debug(
    #                     f"One or more safetensors_metadata keys or values had to be converted to str(). Final safetensors_metadata: {safetensors_metadata}"
    #                 )

    #         # Format is required to enable Accelerate to load the metadata
    #         # otherwise it raises an OSError
    #         safetensors_metadata["format"] = "pt"

    #         # Store the quantization configuration as safetensors metadata
    #         from auto_gptq import __version__

    #         safetensors_metadata["auto_gptq_version"] = str(__version__)
    #         safetensors_metadata["gptq_bits"] = str(self.quantize_config.bits)
    #         safetensors_metadata["gptq_group_size"] = str(self.quantize_config.group_size)
    #         safetensors_metadata["gptq_desc_act"] = str(self.quantize_config.desc_act)
    #         safetensors_metadata["gptq_damp_percent"] = str(self.quantize_config.damp_percent)
    #         safetensors_metadata["gptq_is_marlin_format"] = str(self.quantize_config.is_marlin_format)

    #         safe_save(state_dict, join(save_dir, model_save_name), safetensors_metadata)
    #     else:
    #         model_save_name = model_base_name + ".bin"
    #         torch.save(self.model.state_dict(), join(save_dir, model_save_name))

    #     self.model.config.quantization_config = self.quantize_config.to_dict()
    #     self.model.config.save_pretrained(save_dir)
    #     self.quantize_config.save_pretrained(save_dir)
    #     self.quantize_config.model_name_or_path = save_dir
    #     self.quantize_config.model_file_base_name = model_base_name

    # def save_pretrained(
    #     self,
    #     save_dir: str,
    #     use_safetensors: bool = True,
    #     safetensors_metadata: Optional[Dict[str, str]] = None,
    #     **kwargs,
    # ):
    #     """alias of save_quantized"""
    #     logger.warning("you are using save_pretrained, which will re-direct to save_quantized.")
    #     self.save_quantized(save_dir, use_safetensors, safetensors_metadata)

    # @classmethod
    # def from_pretrained(
    #     cls,
    #     pretrained_model_name_or_path: str,
    #     quantize_config: BaseQuantizeConfig,
    #     max_memory: Optional[dict] = None,
    #     trust_remote_code: bool = False,
    #     torch_dtype: torch.dtype = torch.float16,
    #     **model_init_kwargs,
    # ):
    #     """load un-quantized pretrained model to cpu"""

    #     if not torch.cuda.is_available():
    #         raise EnvironmentError("Load pretrained model to do quantization requires CUDA available.")

    #     def skip(*args, **kwargs):
    #         pass

    #     torch.nn.init.kaiming_uniform_ = skip
    #     torch.nn.init.uniform_ = skip
    #     torch.nn.init.normal_ = skip

    #     # Parameters related to loading from Hugging Face Hub
    #     cache_dir = model_init_kwargs.pop("cache_dir", None)
    #     force_download = model_init_kwargs.pop("force_download", False)
    #     resume_download = model_init_kwargs.pop("resume_download", False)
    #     proxies = model_init_kwargs.pop("proxies", None)
    #     local_files_only = model_init_kwargs.pop("local_files_only", False)
    #     use_auth_token = model_init_kwargs.pop("use_auth_token", None)
    #     revision = model_init_kwargs.pop("revision", None)
    #     subfolder = model_init_kwargs.pop("subfolder", "")
    #     commit_hash = model_init_kwargs.pop("_commit_hash", None)

    #     cached_file_kwargs = {
    #         "cache_dir": cache_dir,
    #         "force_download": force_download,
    #         "proxies": proxies,
    #         "resume_download": resume_download,
    #         "local_files_only": local_files_only,
    #         "use_auth_token": use_auth_token,
    #         "revision": revision,
    #         "subfolder": subfolder,
    #         "_commit_hash": commit_hash,
    #     }

    #     config = AutoConfig.from_pretrained(
    #         pretrained_model_name_or_path, trust_remote_code=True, **cached_file_kwargs
    #     )
    #     if config.model_type not in SUPPORTED_MODELS:
    #         raise TypeError(f"{config.model_type} isn't supported yet.")

    #     # enforce some values despite user specified
    #     model_init_kwargs["torch_dtype"] = torch_dtype
    #     model_init_kwargs["trust_remote_code"] = trust_remote_code
    #     if max_memory:
    #         if "disk" in max_memory:
    #             raise NotImplementedError("disk offload not support yet.")
    #         with accelerate.init_empty_weights():
    #             model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    #         model.tie_weights()

    #         max_memory = accelerate.utils.get_balanced_memory(
    #             model,
    #             max_memory=max_memory,
    #             no_split_module_classes=[cls.layer_type],
    #             dtype=model_init_kwargs["torch_dtype"],
    #             low_zero=False,
    #         )
    #         model_init_kwargs["device_map"] = accelerate.infer_auto_device_map(
    #             model,
    #             max_memory=max_memory,
    #             no_split_module_classes=[cls.layer_type],
    #             dtype=model_init_kwargs["torch_dtype"],
    #         )
    #         model_init_kwargs["low_cpu_mem_usage"] = True

    #         del model
    #     else:
    #         model_init_kwargs["device_map"] = None
    #         model_init_kwargs["low_cpu_mem_usage"] = False

    #     torch.cuda.empty_cache()

    #     merged_kwargs = {**model_init_kwargs, **cached_file_kwargs}
    #     model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, **merged_kwargs)

    #     model_config = model.config.to_dict()
    #     seq_len_keys = ["max_position_embeddings", "seq_length", "n_positions"]
    #     if any(k in model_config for k in seq_len_keys):
    #         for key in seq_len_keys:
    #             if key in model_config:
    #                 model.seqlen = model_config[key]
    #                 break
    #     else:
    #         logger.warning("can't get model's sequence length from model config, will set to 4096.")
    #         model.seqlen = 4096
    #     model.eval()

    #     return cls(model, False, quantize_config)

    @classmethod
    def from_quantized(
        cls,
        model_name_or_path: Optional[str],
        device_map: Optional[Union[str, Dict[str, Union[int, str]]]] = None,
        max_memory: Optional[dict] = None,
        device: Optional[Union[str, int]] = None,
        low_cpu_mem_usage: bool = False,
        use_triton: bool = False,
        use_qigen: bool = False,
        use_marlin: bool = False,
        torch_dtype: Optional[torch.dtype] = None,
        inject_fused_attention: bool = True,
        inject_fused_mlp: bool = True,
        use_cuda_fp16: bool = True,
        quantize_config: Optional[BaseQuantizeConfig] = None,
        model_basename: Optional[str] = None,
        use_safetensors: bool = True,
        trust_remote_code: bool = False,
        warmup_triton: bool = False,
        trainable: bool = False,
        disable_exllama: Optional[bool] = True,
        disable_exllamav2: bool = True,
        **kwargs,
    ):
        # Parameters related to loading from Hugging Face Hub
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", False)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", "")
        commit_hash = kwargs.pop("_commit_hash", None)

        cached_file_kwargs = {
            "cache_dir": cache_dir,
            "force_download": force_download,
            "proxies": proxies,
            "resume_download": resume_download,
            "local_files_only": local_files_only,
            "use_auth_token": use_auth_token,
            "revision": revision,
            "subfolder": subfolder,
            "_raise_exceptions_for_missing_entries": False,
            "_commit_hash": commit_hash,
        }

        # == step1: prepare configs and file names == #
        config = AutoConfig.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            **cached_file_kwargs,
        )

        if config.model_type not in ['llama']:
            raise TypeError(f"{config.model_type} isn't supported yet.")

        if quantize_config is None:
            quantize_config = BaseQuantizeConfig.from_pretrained(model_name_or_path, **cached_file_kwargs, **kwargs)

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
                trainable=trainable,
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
            dtype=torch_dtype,  # This is very hacky but works due to https://github.com/huggingface/accelerate/blob/bd72a5f1a80d5146554458823f8aeda0a9db5297/src/accelerate/utils/modeling.py#L292
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
        # if inject_fused_attention:
        #     if cls.fused_attn_module_type is None:
        #         inject_fused_attention = False
        #         logger.warning(f"{cls.__name__} hasn't fused attention module yet, will skip inject fused attention.")
        #     else:
        #         cls.fused_attn_module_type.inject_to_model(
        #             model,
        #             use_triton=use_triton,
        #             group_size=quantize_config.group_size,
        #             use_cuda_fp16=use_cuda_fp16,
        #             desc_act=quantize_config.desc_act,
        #             trainable=trainable,
        #             bits=quantize_config.bits,
        #             disable_exllama=disable_exllama,
        #             disable_exllamav2=disable_exllamav2,
        #         )
        # if inject_fused_mlp:
        #     if cls.fused_mlp_module_type is None:
        #         inject_fused_mlp = False
        #         logger.warning(f"{cls.__name__} hasn't fused mlp module yet, will skip inject fused mlp.")
        #     else:
        #         cls.fused_mlp_module_type.inject_to_model(model, use_triton=use_triton)

        # Any post-initialization that require device information, for example buffers initialization on device.
        model = autogptq_post_init(model, use_act_order=quantize_config.desc_act)

        model.eval()

        # == step6: (optional) warmup triton == #
        if use_triton and warmup_triton:
            from auto_gptq.nn_modules.qlinear.qlinear_triton import QuantLinear

            QuantLinear.warmup(model, seqlen=model.seqlen)

            # if inject_fused_mlp and cls.fused_mlp_module_type is not None:
            #     cls.fused_mlp_module_type.warmup(model, seqlen=model.seqlen)

        # == step7: make model compatible with peft
        # cls.make_sure_compatible_with_peft(
        #     model,
        #     use_triton,
        #     quantize_config.desc_act,
        #     quantize_config.group_size,
        #     bits=quantize_config.bits,
        #     disable_exllama=disable_exllama,
        #     disable_exllamav2=disable_exllamav2,
        #     use_marlin=use_marlin,
        #     use_qigen=use_qigen,
        # )

        return cls(
            model,
            True,
            quantize_config,
            is_triton_backend=use_triton,
            injected_fused_attention=inject_fused_attention,
            injected_fused_mlp=inject_fused_mlp and use_triton,
            trainable=trainable,
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

    @staticmethod
    def make_sure_compatible_with_peft(
        model: PreTrainedModel,
        use_triton: bool,
        desc_act: bool,
        group_size: int,
        bits: int,
        disable_exllama: bool = True,
        disable_exllamav2: bool = False,
        use_marlin: bool = False,
        use_qigen: bool = False,
    ):
        GeneralQuantLinear.inject_to_model(
            model,
            dynamically_import_QuantLinear(use_triton, desc_act, group_size, bits=bits, disable_exllama=disable_exllama, disable_exllamav2=disable_exllamav2, disable_marlin=not use_marlin, use_qigen=use_qigen),
        )

    # def __getattr__(self, item):
    #     try:
    #         return super().__getattr__(item)
    #     except Exception:
    #         return getattr(self.model, item)


__all__ = ["LlamaGPTQ",]
