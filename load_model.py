import os
from transformers import PreTrainedModel, BitsAndBytesConfig, PretrainedConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from typing import Type
from headed_model import get_multi_head_transformer, HeadedModel
from mlp_head import MLPHead
from headed_config import HeadConfig, create_headed_model_config
from util import find_all_linear_names
import json


def get_from_pretrained(
    base_model_class: Type[PreTrainedModel],
    model_name: str,
    head_configs,
    device_map="auto",
    **kwargs
):
    base_model_config = base_model_class.config_class.from_pretrained(model_name)
    headed_config_class = create_headed_model_config(base_model_class.config_class)
    config = headed_config_class.from_base_class(base_model_config, head_configs)

    model = get_multi_head_transformer(base_model_class)
    model = model.from_pretrained(
        model_name, config=config, device_map=device_map, **kwargs
    )
    return model


def load_trained_qlora_with_heads(
    base_model_class: Type[PreTrainedModel],
    path: str,
    quantization_config: BitsAndBytesConfig,
    only_inference: bool = True,
    device_map="auto",
):
    adapt_config_path = os.path.join(path, "adapter_config.json")
    with open(adapt_config_path, "r") as f:
        base_model_path = json.load(f)["base_model_name_or_path"]


def load_pretrained_qlora(
    base_model_class: Type[PreTrainedModel],
    model_name: str,
    quantization_config: BitsAndBytesConfig,
    lora_config: LoraConfig,
    head_configs: list[HeadConfig],
    untrained_models_save_location: str = None,
    fully_trained_heads: bool = True,
    device_map="auto",
    **kwargs
):
    if untrained_models_save_location is None:
        assert "HOME" in os.environ or "HF_HOME" in os.environ
        untrained_models_save_location = os.path.join(
            os.environ.get(
                "HF_HOME", os.path.join(os.environ["HOME"], ".cache/huggingface/")
            ),
            "untrained_head_models",
        )
    untrained_hash = str(
        hash(str(base_model_class) + model_name + str(hash(tuple(head_configs))))
    )
    untrained_save_location = os.path.join(
        untrained_models_save_location, untrained_hash
    )
    bits = (
        4
        if quantization_config.load_in_4bit
        else 8 if quantization_config.load_in_8bit else 32
    )
    base_model_config: PretrainedConfig = base_model_class.config_class.from_pretrained(
        model_name
    )
    headed_config_class = create_headed_model_config(base_model_class.config_class)
    config = headed_config_class.from_base_class(base_model_config, head_configs)

    model = get_multi_head_transformer(base_model_class)
    if quantization_config.llm_int8_skip_modules is None:
        quantization_config.llm_int8_skip_modules = []
    quantization_config.llm_int8_skip_modules.extend(["MLPHead", "heads"])

    if os.path.exists(untrained_save_location):
        model_name = untrained_save_location

    model: HeadedModel = model.from_pretrained(
        model_name,
        load_in_4bit=quantization_config.load_in_4bit,
        load_in_8bit=quantization_config.load_in_8bit,
        config=config,
        device_map=device_map,
        quantization_config=quantization_config,
        **kwargs
    )

    if not os.path.exists(untrained_save_location):
        model.name_or_path = untrained_save_location

    if lora_config.target_modules is None:
        lora_config.target_modules = find_all_linear_names(
            bits, model, noadd=["MLPHead", "heads"]
        )

    model = prepare_model_for_kbit_training(model)

    model = get_peft_model(model, lora_config)

    if fully_trained_heads:
        head: MLPHead
        for head in model.heads.values():
            head.set_requires_grad(True)
            head.requires_individual_saving = True

    return model
