import torch
import torch.nn as nn
import transformers
import accelerate

from typing import Union
from src.model.QuantLinear import QuantLinear

def get_device(obj: Union[torch.Tensor, nn.Module]):
    if isinstance(obj, torch.Tensor):
        return obj.device
    return next(obj.parameters()).device

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

def make_quant(
    module,
    names,
    bits,
    group_size,
    name="",
    use_triton: bool = True,
    use_cuda_fp16: bool = True,
    desc_act: bool = False,
):
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
            if (not (desc_act) or group_size == -1) and not use_triton:
                new_layer = QuantLinear(
                    bits,
                    group_size,
                    in_features,
                    out_features,
                    True,
                    use_cuda_fp16=use_cuda_fp16,
                    weight_dtype=tmp.weight.dtype,
                )
            else:
                new_layer = QuantLinear(
                    bits,
                    group_size,
                    in_features,
                    out_features,
                    True,
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
        )
