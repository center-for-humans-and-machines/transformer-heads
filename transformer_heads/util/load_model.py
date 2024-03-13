import json
import os
from typing import Type

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig, PretrainedConfig, PreTrainedModel

from transformer_heads.config import HeadConfig, create_headed_model_config
from transformer_heads.model.head import MLPHead
from transformer_heads.model.model import HeadedModel, get_multi_head_transformer

from .model import find_all_linear_names, patch_save_pretrained


def patch_quantization_config(quantization_config: BitsAndBytesConfig):
    if quantization_config.llm_int8_skip_modules is None:
        quantization_config.llm_int8_skip_modules = []
    quantization_config.llm_int8_skip_modules.extend(["MLPHead", "heads"])


def load_headed(
    base_model_class: Type[PreTrainedModel],
    model_name: str,
    head_configs=None,
    head_folder_path=None,
    only_inference: bool = False,
    device_map="auto",
    quantization_config: BitsAndBytesConfig = None,
    **kwargs,
):
    assert head_configs is not None or head_folder_path is not None
    assert head_configs is None or head_folder_path is None
    if head_folder_path is not None:
        with open(os.path.join(head_folder_path, "head_configs.json"), "r") as f:
            head_configs = list(json.load(f).values())
    if quantization_config is None:
        bits = 32
    else:
        patch_quantization_config(quantization_config)
        bits = (
            4
            if quantization_config.load_in_4bit
            else 8
            if quantization_config.load_in_8bit
            else 32
        )
    base_model_config = base_model_class.config_class.from_pretrained(model_name)
    headed_config_class = create_headed_model_config(base_model_class.config_class)
    config = headed_config_class.from_base_class(base_model_config, head_configs)

    model = get_multi_head_transformer(base_model_class)
    model = model.from_pretrained(
        model_name,
        config=config,
        device_map=device_map,
        load_in_4bit=bits == 4,
        load_in_8bit=bits == 8,
        quantization_config=quantization_config,
        **kwargs,
    )
    if quantization_config is not None and bits < 16:
        if not only_inference:
            model = prepare_model_for_kbit_training(model)
        head: MLPHead
        for head in model.heads.values():
            if head_folder_path is not None:
                head.load_from_safetensors(head_folder_path)
            if not only_inference:
                head.set_requires_grad(True)
                head.requires_individual_saving = True
        patch_save_pretrained(model, preserve_old=False)
    return model


def load_qlora_with_heads(
    base_model_class: Type[PreTrainedModel],
    path: str,
    quantization_config: BitsAndBytesConfig,
    only_inference: bool = False,
    fully_trained_heads: bool = True,
    device_map="auto",
    torch_dtype=torch.float32,
    gradient_checkpointing: bool = False,
    **kwargs,
):
    patch_quantization_config(quantization_config)
    adapt_config_path = os.path.join(path, "adapter_config.json")
    with open(adapt_config_path, "r") as f:
        base_model_path = json.load(f)["base_model_name_or_path"]
    with open(os.path.join(path, "head_configs.json"), "r") as f:
        head_configs = list(json.load(f).values())

    base_model_config: PretrainedConfig = base_model_class.config_class.from_pretrained(
        base_model_path
    )

    headed_config_class = create_headed_model_config(base_model_class.config_class)
    config = headed_config_class.from_base_class(base_model_config, head_configs)

    model = get_multi_head_transformer(base_model_class)
    model: HeadedModel = model.from_pretrained(
        base_model_path,
        load_in_4bit=quantization_config.load_in_4bit,
        load_in_8bit=quantization_config.load_in_8bit,
        config=config,
        device_map=device_map,
        quantization_config=quantization_config,
        torch_dtype=torch_dtype,
        **kwargs,
    )

    if not only_inference:
        model: HeadedModel = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=gradient_checkpointing
        )
    model.load_adapter(path, device_map=device_map)

    if not only_inference:
        for name, param in model.named_parameters():
            if "lora" in name:
                param.requires_grad = True
    head: MLPHead
    for head in model.heads.values():
        head.load_from_safetensors(path)
        if not only_inference and fully_trained_heads:
            head.set_requires_grad(True)
            head.requires_individual_saving = True
    patch_save_pretrained(model)
    return model


def create_headed_qlora(
    base_model_class: Type[PreTrainedModel],
    model_name: str,
    quantization_config: BitsAndBytesConfig,
    lora_config: LoraConfig,
    head_configs: list[HeadConfig],
    fully_trained_heads: bool = True,
    device_map="auto",
    gradient_checkpointing: bool = False,
    **kwargs,
):
    patch_quantization_config(quantization_config)
    bits = (
        4
        if quantization_config.load_in_4bit
        else 8
        if quantization_config.load_in_8bit
        else 32
    )
    base_model_config: PretrainedConfig = base_model_class.config_class.from_pretrained(
        model_name
    )
    headed_config_class = create_headed_model_config(base_model_class.config_class)
    config = headed_config_class.from_base_class(base_model_config, head_configs)

    model = get_multi_head_transformer(base_model_class)

    model: HeadedModel = model.from_pretrained(
        model_name,
        load_in_4bit=quantization_config.load_in_4bit,
        load_in_8bit=quantization_config.load_in_8bit,
        config=config,
        device_map=device_map,
        quantization_config=quantization_config,
        **kwargs,
    )

    if lora_config.target_modules is None:
        lora_config.target_modules = find_all_linear_names(bits, model, noadd=["heads"])

    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=gradient_checkpointing
    )

    model = get_peft_model(model, lora_config)

    if fully_trained_heads:
        head: MLPHead
        for head in model.heads.values():
            head.set_requires_grad(True)
            head.requires_individual_saving = True

    patch_save_pretrained(model)
    return model
