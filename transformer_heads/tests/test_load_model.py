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


def check_consistency(outputs1: HeadedModelOutput, outputs2: HeadedModelOutput):
    for key in outputs1.preds_by_head:
        logits1 = outputs1.preds_by_head[key]
        logits2 = outputs2.preds_by_head[key]
        print(key)
        probs1 = torch.softmax(logits1[0], dim=-1)
        probs2 = torch.softmax(logits2[0], dim=-1)
        print(torch.sum(probs1), torch.sum(probs2))
        print(torch.sum(torch.abs(probs1 - probs2)))
        assert probs1.allclose(probs2, atol=1e-4, rtol=1e-3)


def get_test_inputs(device, model_path="mistralai/Mistral-7B-v0.1"):
    tk = AutoTokenizer.from_pretrained(model_path)
    inputs = tk("Paris is the capital of", return_tensors="pt")
    inputs["classes"] = torch.ones_like(inputs["input_ids"])
    inputs["seq"] = torch.tensor(1)
    inputs["regression_hook"] = torch.zeros_like(inputs["input_ids"])
    inputs["lm_head"] = torch.ones_like(inputs["input_ids"])
    inputs.to(device)
    return tk, inputs


def test_load_model(model_path="mistralai/Mistral-7B-v0.1"):
    params = get_model_params(model_path)
    heads[0].num_outputs = params["vocab_size"]
    model = load_headed(params["model_class"], model_path, heads, device_map="cpu")
    print("Loaded headed model successfully!")

    tk, inputs = get_test_inputs(model.device, model_path=model_path)
    outputs: HeadedModelOutput = model(**inputs)
    print("loss_by_head", outputs["loss_by_head"])
    logits = outputs.preds_by_head["lm_head"]
    next_logits = logits[0, -1, :]
    pred_tk = tk.decode(next_logits.argmax().item())
    print("Model prediction:", pred_tk)

    model.save_pretrained("headed_model")
    print("Saved headed model successfully!")
    del model
    model = get_multi_head_transformer(params["model_class"]).from_pretrained(
        "headed_model", device_map="cpu"
    )
    print("Loaded saved headed model successfully!")
    inputs.to(model.device)
    new_outputs: HeadedModelOutput = model(**inputs)
    new_logits = new_outputs.preds_by_head["lm_head"].to(logits.device)
    new_next_logits = logits[0, -1, :]
    pred_tk = tk.decode(new_next_logits.argmax().item())
    print("Model prediction:", pred_tk)
    check_consistency(outputs, new_outputs)


def test_load_quantized(model_path="mistralai/Mistral-7B-v0.1"):
    params = get_model_params(model_path)
    heads[0].num_outputs = params["vocab_size"]
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        load_in_8bit=False,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=torch.float32,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = load_headed(
        params["model_class"],
        model_path,
        heads,
        device_map="cuda",
        quantization_config=quantization_config,
    )
    tk, inputs = get_test_inputs(model.device, model_path=model_path)
    outputs1 = model(**inputs)
    print("loss_by_head", outputs1["loss_by_head"])
    model.save_pretrained("headed_model")
    del model
    model = load_headed(
        params["model_class"],
        model_path,
        head_folder_path="headed_model",
        device_map="cuda",
        quantization_config=quantization_config,
    )
    outputs2 = model(**inputs)
    check_consistency(outputs1, outputs2)


def test_qlora(model_path="mistralai/Mistral-7B-v0.1"):
    params = get_model_params(model_path)
    heads[0].num_outputs = params["vocab_size"]
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        load_in_8bit=False,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=torch.float32,
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
    )
    print_trainable_parameters(model, use_4bit=quantization_config.load_in_4bit)
    print("Loaded headed qlora model successfully!")
    tk, inputs = get_test_inputs(model.device, model_path=model_path)
    print(inputs["input_ids"].dtype)
    outputs: HeadedModelOutput = model(**inputs)
    logits = outputs.preds_by_head["lm_head"]
    next_logits = logits[0, -1, :]
    pred_tk = tk.decode(next_logits.argmax().item())
    print("Model prediction:", pred_tk)
    model.save_pretrained("headed_model_qlora")
    print("Saved headed qlora model successfully!")
    del model
    model = load_lora_with_heads(
        params["model_class"],
        "headed_model_qlora",
        quantization_config,
        device_map="cuda",
    )
    print_trainable_parameters(model, use_4bit=quantization_config.load_in_4bit)
    print("Loaded saved headed qlora model successfully!")
    print(inputs["input_ids"].dtype)
    new_outputs: HeadedModelOutput = model(**inputs)
    new_logits = new_outputs.preds_by_head["lm_head"]
    new_next_logits = new_logits[0, -1, :]
    pred_tk = tk.decode(new_next_logits.argmax().item())
    print("Model prediction:", pred_tk)
    check_consistency(outputs, new_outputs)


if __name__ == "__main__":
    Fire()
