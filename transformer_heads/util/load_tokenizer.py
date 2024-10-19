"""
This module provides a function to load a tokenizer from a adapter model path.

Functions:
    load_tokenizer: Loads a tokenizer from a given model path.
"""

import json
import os

from transformers import AutoTokenizer


def load_tokenizer(model_path):
    """
    Loads a tokenizer from a given adapter model path.

    The function first checks if an adapter configuration file exists in the model path. If it does, the function
    reads the base model path from the configuration file and loads the tokenizer from the base model path.
    If the adapter configuration file does not exist, the function raises a FileNotFoundError.

    Args:
        model_path (str): The path to the model.

    Returns:
        PreTrainedTokenizer: The loaded tokenizer.

    Raises:
        FileNotFoundError: If the adapter configuration file does not exist in the model path.
    """
    adapt_config_path = os.path.join(model_path, "adapter_config.json")
    if os.path.exists(adapt_config_path):
        with open(adapt_config_path, "r") as f:
            base_model_path = json.load(f)["base_model_name_or_path"]
    else:
        raise FileNotFoundError(
            f"Adapter config file not found in {model_path}. If you are not trying to load from an adapter_model, just load the tokenizer with AutoTokenizer.from_pretrained"
        )
    tk = AutoTokenizer.from_pretrained(base_model_path)
    if tk.pad_token is None:
        tk.pad_token = tk.eos_token
    return tk
