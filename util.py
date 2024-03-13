import torch
import bitsandbytes as bnb
from collections import defaultdict


def patch_state_dict(state_dict):
    return {
        key: value if value.dim() > 0 else torch.unsqueeze(value, 0)
        for key, value in state_dict.items()
    }


def find_all_linear_names(bits, model, noadd=[]):
    # Source https://github.com/artidoro/qlora/blob/main/qlora.py#L248
    cls = (
        bnb.nn.Linear4bit
        if bits == 4
        else (bnb.nn.Linear8bitLt if bits == 8 else torch.nn.Linear)
    )
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            for n in names:
                if n in noadd:
                    break
            else:
                lora_module_names.add(names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def print_trainable_parameters(model, use_4bit=False):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    params_by_dtype = defaultdict(int)
    trainable_params_by_dtype = defaultdict(int)
    for name, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        params_by_dtype[param.dtype] += num_params
        if param.requires_grad:
            trainable_params_by_dtype[param.dtype] += num_params
            trainable_params += num_params
    if use_4bit:
        trainable_params /= 2
    print(
        f"all params: {all_param} || trainable params: {trainable_params} || trainable%: {100 * trainable_params / all_param}"
    )
    print("params by dtype:", params_by_dtype)
    print("trainable params by dtype:", trainable_params_by_dtype)
