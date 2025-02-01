import gc

import torch
from fire import Fire
from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
)

from transformer_heads.config import HeadConfig
from transformer_heads.model.model import get_multi_head_transformer
from transformer_heads.output import HeadedModelOutput
from transformer_heads.util.helpers import get_model_params
from transformer_heads.util.load_model import (
    create_headed_qlora,
    load_headed,
    load_lora_with_heads,
)
from transformer_heads.util.model import compare_all_params, print_trainable_parameters


heads = [
    HeadConfig(
        name="value_head",
        layer_hook=-1,
        in_size=4096,
        hidden_size=1024,
        num_layers=2,
        output_activation="linear",
        is_causal_lm=False,
        loss_fct="masked_mse",
        num_outputs=1,
        is_regression=False,
        output_bias=False,
        target="values",
    ),
]


def get_training_data(device="cuda"):
    data = {}
    input = torch.tensor([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]], device=device)
    labels = torch.tensor(
        [[0, 1, 0, 0.2, -100], [0.2, 0.3, 0.1, -100, -100]], device=device
    ).unsqueeze(-1)
    data["input_ids"] = input
    data["values"] = labels
    return data


def test_gradients(model_path="mistralai/Mistral-7B-v0.1"):
    torch.autograd.set_detect_anomaly(True)
    torch_dtype = torch.bfloat16
    params = get_model_params(model_path)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        load_in_8bit=False,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    lora_config = LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules=None,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = create_headed_qlora(
        params["model_class"],
        model_path,
        quantization_config=quantization_config,
        lora_config=lora_config,
        head_configs=heads,
        device_map="cuda",
        torch_dtype=torch_dtype,
        attn_implementation="flash_attention_2",
    )
    inputs = get_training_data(device=model.device)

    outputs: HeadedModelOutput = model(**inputs)

    print("Loss:", outputs.loss)

    outputs.loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, torch.sum(param.grad))


if __name__ == "__main__":
    Fire()
