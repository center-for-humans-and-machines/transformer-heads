import torch
from fire import Fire
from peft import LoraConfig
from transformers import AutoTokenizer, BitsAndBytesConfig, MistralForCausalLM

from transformer_heads.config import HeadConfig
from transformer_heads.model.model import get_multi_head_transformer
from transformer_heads.output import HeadedModelOutput
from transformer_heads.util.load_model import (
    create_headed_qlora,
    load_headed,
    load_lora_with_heads,
)
from transformer_heads.util.helpers import get_model_params
from transformer_heads.util.model import print_trainable_parameters

heads = [
    HeadConfig(
        name="lm_head",
        layer_hook=-1,
        in_size=4096,
        hidden_size=0,
        num_layers=1,
        output_activation="linear",
        is_causal_lm=True,
        loss_fct="cross_entropy",
        num_outputs=32000,
        is_regression=False,
        output_bias=False,
    ),
    HeadConfig(
        name="classification_hook",
        layer_hook=-4,
        in_size=4096,
        hidden_size=1024,
        num_layers=2,
        output_activation="linear",
        is_causal_lm=False,
        loss_fct="cross_entropy",
        num_outputs=2,
        is_regression=False,
        output_bias=False,
        target="classes",
    ),
    HeadConfig(
        name="classify_seq",
        layer_hook=-4,
        in_size=4096,
        hidden_size=1024,
        num_layers=2,
        output_activation="linear",
        is_causal_lm=False,
        loss_fct="cross_entropy",
        num_outputs=2,
        pred_for_sequence=True,
        is_regression=False,
        output_bias=False,
        target="seq",
    ),
    HeadConfig(
        name="regression_hook",
        layer_hook=-6,
        in_size=4096,
        hidden_size=0,
        num_layers=1,
        output_activation="linear",
        is_causal_lm=False,
        loss_fct="mse",
        num_outputs=1,
        is_regression=True,
        output_bias=False,
    ),
]


def get_test_inputs(device, model_path="mistralai/Mistral-7B-v0.1"):
    tk = AutoTokenizer.from_pretrained(model_path)
    if tk.pad_token_id is None:
        tk.pad_token = tk.eos_token
    inputs = tk(
        ["Paris is the capital of", "I am the"], return_tensors="pt", padding=True
    )
    inputs["classes"] = torch.randint(
        low=0, high=2, size=(inputs["input_ids"].size(0), inputs["input_ids"].size(1))
    )
    inputs["seq"] = torch.randint(low=0, high=2, size=(inputs["input_ids"].size(0),))
    inputs["regression_hook"] = torch.rand(
        (inputs["input_ids"].size(0), inputs["input_ids"].size(1))
    )
    inputs["lm_head"] = torch.randint(
        low=0,
        high=32000,
        size=(inputs["input_ids"].size(0), inputs["input_ids"].size(1)),
    )
    inputs.to(device)
    return tk, inputs


def test_adaptive_loss(model_path="mistralai/Mistral-7B-v0.1"):
    model_params = get_model_params(model_path)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = load_headed(
        model_params["model_class"],
        model_path,
        heads,
        device_map="cuda",
        quantization_config=quantization_config,
    )
    model.set_adaptive_loss(True)
    model.train()

    for _ in range(20):
        tk, inputs = get_test_inputs(model.device, model_path)
        outputs = model(**inputs)
        print(
            f"\nLoss: {outputs['loss']}\nloss_type:{type(outputs['loss'])}\nloss_by_head: {outputs['loss_by_head']}\nadapted_loss_by_head: {outputs['adapted_loss_by_head']}"
        )
        print(f"Loss requires grad: {outputs['loss'].requires_grad}")


def test_loss(model_path="mistralai/Mistral-7B-v0.1"):
    model_params = get_model_params(model_path)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = load_headed(
        model_params["model_class"],
        model_path,
        heads,
        device_map="cuda",
        quantization_config=quantization_config,
    )

    tk, inputs = get_test_inputs(model.device, model_path)

    outputs = model(**inputs)

    print(outputs["loss"], outputs["loss_by_head"])


if __name__ == "__main__":
    Fire()
