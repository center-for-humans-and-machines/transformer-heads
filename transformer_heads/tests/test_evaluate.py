from transformer_heads import create_headed_qlora, load_lora_with_heads
import fire
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    MistralForCausalLM,
    Trainer,
    BitsAndBytesConfig,
    TrainingArguments,
    GPT2Model,
    GPT2LMHeadModel,
)
from transformer_heads.util.helpers import DataCollatorWithPadding, get_model_params
from peft import LoraConfig
from transformer_heads.config import HeadConfig
from transformer_heads.util.model import print_trainable_parameters
from transformer_heads.util.evaluate import (
    evaluate_head_wise,
)
import torch
import pandas as pd


def test_evaluate():
    model_path = "mistralai/Mistral-7B-v0.1"
    model_params = get_model_params(model_path)
    model_class = model_params["model_class"]
    hidden_size = model_params["hidden_size"]
    vocab_size = model_params["vocab_size"]
    head_configs = [
        HeadConfig(
            name=f"sentiment_head",
            layer_hook=-4,
            in_size=hidden_size,
            output_activation="linear",
            pred_for_sequence=True,
            loss_fct="cross_entropy",
            num_outputs=2,
            loss_weight=2.0,
        ),
        HeadConfig(
            name=f"causal_lm",
            layer_hook=-1,
            in_size=hidden_size,
            output_activation="linear",
            is_causal_lm=True,
            loss_fct="cross_entropy",
            num_outputs=vocab_size,
            is_regression=False,
            output_bias=False,
            loss_weight=1.0,
        ),
        HeadConfig(
            name=f"alphabet_regression",
            layer_hook=-7,
            in_size=hidden_size,
            output_activation="linear",
            is_causal_lm=False,
            pred_for_sequence=True,
            loss_fct="mse",
            num_outputs=26,  # 26 letters in the alphabet
            is_regression=True,
            loss_weight=0.002,
        ),
        HeadConfig(
            name=f"num_tokens_regression",
            layer_hook=-7,
            hidden_size=128,  # MLP hidden size
            num_layers=3,  # 2 hidden layers in MLP
            in_size=hidden_size,
            output_activation="linear",
            is_causal_lm=False,
            pred_for_sequence=False,
            loss_fct="mse",
            num_outputs=1,
            is_regression=True,
            loss_weight=0.0002,
        ),
        HeadConfig(
            name=f"lm_head",  # Let's also keep the original lm head for comparison
            layer_hook=-1,
            in_size=hidden_size,
            output_activation="linear",
            is_causal_lm=True,
            pred_for_sequence=False,
            loss_fct="cross_entropy",
            num_outputs=vocab_size,
            is_regression=False,
            trainable=False,  # Keep it in it's pretrained state
        ),
    ]

    dd = load_dataset("imdb")
    for split in dd.keys():
        dd[split] = dd[split].train_test_split(test_size=0.05)["test"]

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    def processing_function(examples):
        out = tokenizer(examples["text"], padding=False, truncation=True)
        out["sentiment_head"] = examples["label"]
        out["causal_lm"] = out["lm_head"] = out["input_ids"].copy()
        out["num_tokens_regression"] = [
            list(map(float, range(len(ids) - 1, -1, -1))) for ids in out["input_ids"]
        ]
        out["alphabet_regression"] = [
            [
                float(text.count(x) + text.count(x.upper()))
                for x in "abcdefghijklmnopqrstuvwxyz"
            ]
            for text in examples["text"]
        ]
        return out

    for split in dd.keys():
        dd[split] = dd[split].filter(function=lambda example: len(example["text"]) > 10)
        dd[split] = dd[split].shuffle()
        dd[split] = dd[split].map(processing_function, batched=True)

    dd.set_format(
        type="torch",
        columns=["input_ids", "attention_mask"] + [x.name for x in head_configs],
    )
    for split in dd.keys():
        dd[split] = dd[split].remove_columns(["text", "label"])
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
        r=32,
        lora_alpha=16,
        target_modules=None,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = create_headed_qlora(
        base_model_class=model_class,
        model_name=model_path,
        quantization_config=quantization_config,
        lora_config=lora_config,
        head_configs=head_configs,
        fully_trained_heads=True,
        device_map={"": torch.cuda.current_device()},
        gradient_checkpointing=True,
    )

    collator = DataCollatorWithPadding(
        feature_name_to_padding_value={
            "input_ids": tokenizer.pad_token_id,
            "attention_mask": 0,
            "causal_lm": -100,
            "lm_head": -100,
            "num_tokens_regression": 0,
        }
    )

    print(evaluate_head_wise(model, dd["test"], collator, epochs=0.04, batch_size=4))


if __name__ == "__main__":
    fire.Fire(test_evaluate)
