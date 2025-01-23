"""
This module provides helper functions and classes for handling data and models in a language model training and evaluation pipeline.
It includes a data collator for padding sequences and a function for getting model parameters based on the model type.

Classes:
    DataCollatorWithPadding: A data collator that pads sequences to the same length.

Functions:
    get_model_params(model_path: str): Get the parameters of a model based on its type.
"""

from dataclasses import dataclass
from math import sqrt
from typing import Any, Dict, List

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoConfig


@dataclass
class DataCollatorWithPadding:
    """
    A data collator that pads sequences to the same length.

    Attributes:
        feature_name_to_padding_value (dict[str, int]): A dictionary mapping feature names to their padding values.

    Methods:
        __call__(features: List[Dict[str, Any]]) -> Dict[str, Any]: Pad the sequences in the features to the same length.
    """

    feature_name_to_padding_value: dict[str, int | float]

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Pad the sequences in the features to the same length.

        Args:
            features (List[Dict[str, Any]]): A list of features, where each feature is a dictionary mapping feature names to sequences.

        Returns:
            Dict[str, Any]: A dictionary mapping feature names to padded sequences.
        """
        batch = dict()
        for key, value in self.feature_name_to_padding_value.items():
            batch[key] = pad_sequence(
                [feature[key].clone().detach() for feature in features],
                batch_first=True,
                padding_value=value,
            )
        for key in features[0].keys():
            if key not in self.feature_name_to_padding_value:
                batch[key] = torch.stack(
                    [feature[key].clone().detach() for feature in features]
                )
        return batch


def get_model_params(model_path: str):
    """
    Get the parameters of a model based on its type.

    Args:
        model_path (str): The name of the huggingface model.

    Returns:
        dict: A dictionary containing the model class, hidden size, and vocab size.

    Raises:
        ValueError: If the model type is unknown.
    """
    cfg = AutoConfig.from_pretrained(model_path).to_dict()
    cfg["model_class"] = getattr(
        __import__("transformers", fromlist=[cfg["architectures"][0]]),
        cfg["architectures"][0],
    )
    if model_path == "gpt2" and "hidden_size" not in cfg:
        cfg["hidden_size"] = 768
    return cfg


def compare_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare two dictionaries with the same keys and return all keys with differing values.

    Args:
        dict1 (Dict[str, Any]): The first dictionary.
        dict2 (Dict[str, Any]): The second dictionary.

    Returns:
        Dict[str, Any]: A dictionary containing the keys with differing values and their values from both dictionaries.
    """
    differences = {}
    for key in dict1:
        if isinstance(dict1[key], torch.Tensor):
            diff = not torch.equal(dict1[key], dict2[key])
        elif isinstance(dict1[key], np.ndarray):
            diff = not np.array_equal(dict1[key], dict2[key])
        else:
            diff = dict1[key] != dict2[key]
        if diff:
            differences[key] = {"dict1": dict1[key], "dict2": dict2[key]}
    return differences


def compare_objects(obj1: Any, obj2: Any, pre=""):
    diffs = compare_dicts(obj1.__dict__, obj2.__dict__)
    for key, value in diffs.items():
        print(pre, key)
        if hasattr(value["dict1"], "__dict__") and hasattr(value["dict2"], "__dict__"):
            compare_objects(value["dict1"], value["dict2"], pre + "\t")


class Welfords:
    def __init__(self):
        self.count: int = 0
        self.mean: float = 0.0
        self.M2: float = 0.0

    def update(self, new_value):
        self.count += 1
        delta = new_value - self.mean
        self.mean += delta / self.count
        delta2 = new_value - self.mean
        self.M2 += delta * delta2

    @property
    def std(self) -> float:
        return sqrt(self.M2 / self.count) if self.count > 1 else 0.0
