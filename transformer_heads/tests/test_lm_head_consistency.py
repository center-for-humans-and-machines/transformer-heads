from gc import collect

import torch
from fire import Fire
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
    PreTrainedModel,
)
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutput

from transformer_heads.config import HeadConfig
from transformer_heads.util.helpers import get_model_params
from transformer_heads.util.load_model import load_headed
from transformer_heads.util.model import compare_all_params

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


def check_hidden_consistency(model_path="mistralai/Mistral-7B-v0.1"):
    tk = AutoTokenizer.from_pretrained(model_path)

    compute_dtype = torch.bfloat16

    input_ids = tk.encode("Hello, my dog is cute", return_tensors="pt").to("cuda")
    quantization_config = None
    params = get_model_params(model_path)
    th_model = load_headed(
        params["model_class"],
        model_path,
        heads,
        device_map="cuda",
        quantization_config=quantization_config,
        torch_dtype=compute_dtype,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map="cuda",
        torch_dtype=compute_dtype,
    )

    th_model.eval()
    base_model.eval()

    for name, param in th_model.named_parameters():
        param.requires_grad = False
    for name, param in base_model.named_parameters():
        param.requires_grad = False

    with torch.no_grad():
        compare_all_params(th_model.model, base_model.model)

        th_output: BaseModelOutputWithPast = th_model.model(
            input_ids, output_hidden_states=True
        )
        base_output: BaseModelOutputWithPast = base_model.model(
            input_ids, output_hidden_states=True
        )

        assert torch.equal(th_output.last_hidden_state, base_output.last_hidden_state)


def check_forward_consistency(model_path="mistralai/Mistral-7B-v0.1"):
    tk = AutoTokenizer.from_pretrained(model_path)

    compute_dtype = torch.float32

    input_ids = tk.encode("Hello, my dog is cute", return_tensors="pt").to("cuda")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        load_in_8bit=False,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    params = get_model_params(model_path)
    th_model = load_headed(
        params["model_class"],
        model_path,
        heads,
        device_map="cuda",
        quantization_config=quantization_config,
        torch_dtype=compute_dtype,
    )
    with torch.no_grad():
        th_output = th_model(input_ids, output_hidden_states=True)
        th_preds = th_output.preds_by_head["lm_head"]
    # del th_model
    # del th_output
    # collect()
    # torch.cuda.empty_cache()

    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map="cuda",
        torch_dtype=compute_dtype,
    )

    compare_all_params(th_model, base_model)

    with torch.no_grad():
        base_output: CausalLMOutput = base_model(input_ids, output_hidden_states=True)

        base_preds = base_output.logits
        print(th_preds.dtype, base_preds.dtype)

        diff = th_preds - base_preds

        print("th max", th_preds.max().item())
        print("th min", th_preds.min().item())
        print("Max diff:", diff.abs().max().item())
        print("Mean diff:", diff.abs().mean().item())


def check_generate_consistency(model_path="mistralai/Mistral-7B-v0.1"):
    torch_dtype = torch.bfloat16

    generation_config = GenerationConfig(do_sample=False, max_new_tokens=50)
    tk = AutoTokenizer.from_pretrained(model_path)

    input_ids = tk.encode("Hello, my dog is cute", return_tensors="pt").to("cuda")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        load_in_8bit=False,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    params = get_model_params(model_path)
    th_model = load_headed(
        params["model_class"],
        model_path,
        heads,
        device_map="cuda",
        quantization_config=quantization_config,
        torch_dtype=torch_dtype,
    )
    print("loaded model sucessfully")
    with torch.no_grad():
        some_out = th_model(input_ids)
        print("managed some out")
        th_output = th_model.generate(input_ids, generation_config=generation_config)
        th_preds = th_output.sequences

    del th_model
    del th_output
    collect()
    torch.cuda.empty_cache()

    base_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map="cuda",
        torch_dtype=torch_dtype,
    )

    with torch.no_grad():
        base_preds = base_model.generate(input_ids, generation_config=generation_config)

        print(base_preds == th_preds)


if __name__ == "__main__":
    Fire()
