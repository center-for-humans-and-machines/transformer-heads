from transformers import MistralForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from load_model import get_from_pretrained, load_pretrained_qlora
import torch
from headed_model import get_multi_head_transformer
from headed_output import HeadedModelOutput
from headed_config import HeadConfig
from fire import Fire

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


def test_load_model():
    model = get_from_pretrained(
        MistralForCausalLM, "mistralai/Mistral-7B-v0.1", heads, device_map="cpu"
    )
    print("Loaded headed Mistral model successfully!")

    tk = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    inputs = tk("Paris is the capital of", return_tensors="pt")
    inputs.to(model.device)
    outputs: HeadedModelOutput = model(**inputs)
    logits = outputs.logits_by_head["lm_head"]
    next_logits = logits[0, -1, :]
    pred_tk = tk.decode(next_logits.argmax().item())
    print("Model prediction:", pred_tk)

    model.save_pretrained("mistral_headed")
    print("Saved headed Mistral model successfully!")
    del model
    model = get_multi_head_transformer(MistralForCausalLM).from_pretrained(
        "mistral_headed", device_map="cpu"
    )
    print("Loaded saved headed Mistral model successfully!")
    inputs.to(model.device)
    outputs: HeadedModelOutput = model(**inputs)
    new_logits = outputs.logits_by_head["lm_head"].to(logits.device)
    new_next_logits = logits[0, -1, :]
    pred_tk = tk.decode(new_next_logits.argmax().item())
    print("Model prediction:", pred_tk)
    print(torch.sum(new_logits - logits))
    assert new_logits.equal(logits)


def test_quantized():
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
    qlora_model = load_pretrained_qlora(
        MistralForCausalLM,
        "mistralai/Mistral-7B-v0.1",
        quantization_config=quantization_config,
        lora_config=lora_config,
        head_configs=heads,
        device_map="cpu",
    )
    print("Loaded headed qlora Mistral model successfully!")


if __name__ == "__main__":
    Fire()
