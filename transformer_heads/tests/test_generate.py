from transformer_heads.config import HeadConfig
from transformer_heads.util.helpers import get_model_params
from transformer_heads import create_headed_qlora
from transformer_heads.output import HeadedModelGenerateOutput
from transformers import (
    BitsAndBytesConfig,
    AutoTokenizer,
    GenerationConfig,
)
from peft import LoraConfig
import fire
import torch

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
        num_outputs=128256,
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


def _get_model_tk():
    model_path = "meta-llama/Meta-Llama-3-8B"
    model_params = get_model_params(model_path)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        load_in_8bit=False,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=torch.bfloat16,
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
        base_model_class=model_params["model_class"],
        head_configs=heads,
        model_name=model_path,
        quantization_config=quantization_config,
        lora_config=lora_config,
        device_map="cuda",
    )
    tk = AutoTokenizer.from_pretrained(model_path)
    return model, tk


def test_greedy_generation():
    model, tk = _get_model_tk()

    in_text = "God is dead and we"

    inputs = tk.encode(in_text, return_tensors="pt")

    gen_config = GenerationConfig(do_sample=False, max_new_tokens=10)

    generation_output: HeadedModelGenerateOutput = model.generate(
        inputs.to(model.device), generation_config=gen_config
    )

    print(tk.decode(generation_output.sequences[0].tolist()))

    print(
        generation_output.logprobs.shape,
        generation_output.head_outputs["regression_hook"].shape,
    )


def test_sample_generation():
    model, tk = _get_model_tk()
    in_text = "God is dead and we"

    inputs = tk.encode(in_text, return_tensors="pt")

    gen_config = GenerationConfig(
        do_sample=True, max_new_tokens=10, temperature=1.0, top_k=50
    )

    generation_output: HeadedModelGenerateOutput = model.generate(
        inputs.to(model.device), generation_config=gen_config
    )

    print(tk.decode(generation_output.sequences[0].tolist()))

    print(
        generation_output.logprobs.shape,
        generation_output.head_outputs["regression_hook"].shape,
    )

    print(generation_output.logprobs)


if __name__ == "__main__":
    fire.Fire()
