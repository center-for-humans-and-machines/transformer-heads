import torch
import bitsandbytes as bnb


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
