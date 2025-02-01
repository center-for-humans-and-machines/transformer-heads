import torch
from peft.tuners.lora import LoraLayer
from transformers import PreTrainedModel

from transformer_heads.model.head import MLPHead
from transformer_heads.model.model import HeadedModel


def set_compute_dtype(model: PreTrainedModel, compute_dtype: torch.dtype):
    loaded_in_kbit = getattr(model, "is_loaded_in_8bit", False) or getattr(
        model, "is_loaded_in_4bit", False
    )
    if loaded_in_kbit:
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                module.to(compute_dtype)
            elif "norm" in name:
                module.to(torch.float32)
            elif "head" in name or "embed_tokens" in name:
                if hasattr(module, "weight") and module.weight.dtype == torch.float32:
                    module.to(compute_dtype)
    else:
        for name, parameter in model.named_parameters():
            parameter.to(compute_dtype)


def disable_requires_grad(
    model: PreTrainedModel,
):
    for name, param in model.named_parameters():
        # freeze base model's layers
        param.requires_grad = False


def set_requires_grad(model: HeadedModel, fully_train_heads: bool = True):
    for name, param in model.named_parameters():
        if "lora" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    if fully_train_heads:
        head: MLPHead
        for head in model.heads.values():
            if head.trainable:
                head.set_requires_grad(True)
                head.requires_individual_saving = True
        if model.lm_head is not None and model.lm_head_config.trainable:
            model.lm_head.requires_grad_(True)
