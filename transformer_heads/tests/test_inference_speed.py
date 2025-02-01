import gc
import time
from itertools import product

import fire
import pandas
import torch
from peft.tuners.lora import LoraConfig
from transformers import AutoTokenizer, BitsAndBytesConfig, GenerationConfig

from transformer_heads.config import HeadConfig
from transformer_heads.util.helpers import get_model_params
from transformer_heads.util.load_model import create_headed_qlora, load_headed

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
        is_regression=False,
        output_bias=False,
        num_outputs=32000,
    )
]


def test_speed(
    model_path="mistralai/Mistral-7B-v0.1",
    n_examples=8,
    attn_implementation="eager",
    compute_dtype=torch.float32,
    quantize=False,
    only_inference=False,
    qlora=True,
    generate=True,
    batch_size=1,
    input_size=20,
):
    tk = AutoTokenizer.from_pretrained(model_path)

    # input_ids = tk.encode("Hello, my dog is cute", return_tensors="pt").to("cuda")
    input_ids = torch.tensor([[1] * input_size], dtype=torch.int64).to("cuda")

    generation_config = GenerationConfig(do_sample=True, max_new_tokens=10)
    lora_config = LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules=None,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    if quantize:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            load_in_8bit=False,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        quantization_config = None

    params = get_model_params(model_path)

    if qlora:
        th_model = create_headed_qlora(
            params["model_class"],
            model_path,
            quantization_config=quantization_config,
            lora_config=lora_config,
            head_configs=heads,
            device_map="cuda",
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_implementation,
        )
    else:
        th_model = load_headed(
            params["model_class"],
            model_path,
            heads,
            device_map="cuda",
            quantization_config=quantization_config,
            torch_dtype=compute_dtype,
            attn_implementation=attn_implementation,
            only_inference=only_inference,
        )

    input_ids = input_ids.repeat(n_examples, 1)
    torch.cuda.synchronize()
    with torch.no_grad():
        start = time.perf_counter()
        for i in range(n_examples // batch_size):
            my_input_ids = input_ids[i * batch_size : (i + 1) * batch_size]
            if generate:
                th_model.generate(my_input_ids, generation_config=generation_config)
            else:
                th_output = th_model(my_input_ids)
        end = time.perf_counter()
    return end - start


def test_all_conditions(out_path=None):
    param_map = {
        "compute_dtype": [torch.float32, torch.bfloat16],
        "attn_implementation": ["eager", "flash_attention_2"],
        "quantize": [True, False],
        # "only_inference": [True, False],
        # "qlora": [True, False],
        "generate": [True, False],
        "batch_size": [1, 8],
        "input_size": [20, 2048],
    }

    param_map = {
        "batch_size": [1, 8],
        "input_size": [20, 2048],
        "compute_dtype": [torch.float32, torch.bfloat16],
    }

    invalid_combinations = [
        {"compute_dtype": torch.float32, "attn_implementation": "flash_attention_2"},
        {"only_inference": True, "qlora": True},
        # {"batch_size": 8, "input_size": 2048},
    ]

    # Burn run
    test_speed()

    results = []
    keys, values = zip(*param_map.items())
    for combination in product(*values):
        params = dict(zip(keys, combination))
        if any(
            all(k in params and params[k] == v for k, v in invalid.items())
            for invalid in invalid_combinations
        ):
            continue
        try:
            duration = test_speed(**params)
            params["duration"] = duration
            results.append(params)
        except Exception as e:
            print(f"Failed with {params}: {e}")
            continue
        print(
            f"Duration: {duration:.4f}s | "
            + " | ".join(f"{key}: {value}" for key, value in params.items())
        )
        gc.collect()
        torch.cuda.empty_cache()

    df = pandas.DataFrame(results)
    print(df)
    if out_path is not None:
        df.to_csv(out_path)


if __name__ == "__main__":
    fire.Fire()
