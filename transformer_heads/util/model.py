"""
This module contains utility functions for handling and modifying the state of a model, finding all linear names in a model, 
printing the number of trainable parameters in the model, and patching the save_pretrained method of a model.

Functions:
    patch_state_dict(state_dict: Dict):
        Patch a state_dict to fix problems with zero-dimensional tensors.

    find_all_linear_names(bits: int, model: torch.nn.Module, noadd: List[str] = []):
        Find all linear modules in a model.

    print_trainable_parameters(model: torch.nn.Module, use_4bit: bool = False):
        Print some information about the trainable parameters off a model.

    patch_save_pretrained(model: torch.nn.Module, preserve_old: bool = True):
        Patch the save_pretrained method of a model to save heads and head configurations.
"""

import json
import os
from collections import defaultdict
from os import PathLike
from types import MethodType
from typing import TYPE_CHECKING, Any, Callable, Dict

import bitsandbytes as bnb
import torch

if TYPE_CHECKING:
    from transformer_heads.model.head import MLPHead


def patch_state_dict(state_dict):
    """
    Patch a state_dict to fix problems with zero-dimensional tensors.

    Args:
        state_dict (Dict): The state dictionary of a model.

    Returns:
        Dict: The modified state dictionary.
    """
    return {
        key: value if value.dim() > 0 else torch.unsqueeze(value, 0)
        for key, value in state_dict.items()
    }


def find_all_linear_names(bits, model, noadd=[]):
    """
    Find all linear modules in a model.

    Args:
        bits (int): The number of bits used in quantization. (set to 32 for unquantized model)
        model (torch.nn.Module): The model to find linear names in.
        noadd (List[str], optional): A list of names to exclude. Defaults to [].

    Returns:
        List[str]: A list of all linear names in the model.
    """
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

    # if "lm_head" in lora_module_names:  # needed for 16-bit
    #    lora_module_names.remove("lm_head")
    return list(lora_module_names)


def print_trainable_parameters(model, use_4bit=False):
    """
    Print some information about the trainable parameters off a model.

    Args:
        model (torch.nn.Module): The model to print the number of trainable parameters for.
        use_4bit (bool, optional): Whether 4-bit quantization is used. Defaults to False.
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


def patch_save_pretrained(model, preserve_old: bool = True):
    """
    Patch the save_pretrained method of a model to save heads and head configurations.

    Args:
        model (torch.nn.Module): The model to patch the save_pretrained method for.
        preserve_old (bool, optional): Whether to preserve (and call) the old save_pretrained method. Defaults to True.
    """

    def save_pretrained(
        self,
        save_directory: str | PathLike,
        is_main_process: bool = True,
        state_dict: Dict | None = None,
        save_function: Callable[..., Any] = torch.save,
        push_to_hub: bool = False,
        max_shard_size: int | str = "5GB",
        safe_serialization: bool = True,
        variant: str | None = None,
        token: str | bool | None = None,
        save_peft_format: bool = True,
        **kwargs,
    ):
        os.makedirs(save_directory, exist_ok=True)
        self.old_save_pretrained(
            save_directory=save_directory,
            is_main_process=is_main_process,
            state_dict=state_dict,
            save_function=save_function,
            push_to_hub=push_to_hub,
            max_shard_size=max_shard_size,
            safe_serialization=safe_serialization,
            variant=variant,
            token=token,
            save_peft_format=save_peft_format,
            **kwargs,
        )
        head: MLPHead
        for head in self.heads.values():
            if head.requires_individual_saving:
                head.save_to_safetensors(save_directory)
        with open(os.path.join(save_directory, "head_configs.json"), "w") as f:
            json.dump(self.head_configs, f)

    if preserve_old:
        model.old_save_pretrained = model.save_pretrained
    else:
        model.old_save_pretrained = MethodType(lambda *args, **kwargs: None, model)
    model.save_pretrained = MethodType(save_pretrained, model)
