"""
This module contains functions for evaluating a HeadedModel, a type of model that has multiple "heads" for different tasks.
It provides functions to compute the model loss for each of its heads, get predictions from the model, and get the top n predictions for a given text.

Functions:
    evaluate_head_wise(model: HeadedModel, ds: Dataset, collator=None, batch_size=8, epochs=1) -> tuple[int, dict[str, int]]:
        Compute the model loss for each of its heads.

    get_some_preds(model, ds, tokenizer, n=5, classification=True) -> tuple[list[str], dict[str, list[int]], dict[str, list[int]]]:
        Get predictions from the model.

    get_top_n_preds(n: int, model: HeadedModel, text: str, tokenizer: PreTrainedTokenizer):
        Get the top n predictions for a given text. Use for models with causal language modeling heads.
"""

from collections import defaultdict

import numpy as np
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from transformer_heads.model.model import HeadedModel
from transformer_heads.output import HeadedModelOutput


@torch.inference_mode()
def evaluate_head_wise(
    model: HeadedModel, ds: Dataset, collator=None, batch_size=8, epochs=1
) -> tuple[int, dict[str, int]]:
    """
    Compute the model loss for each of its heads.

    Args:
        model (HeadedModel): The model to be evaluated.
        ds (Dataset): The dataset to be used for evaluation.
        collator (callable, optional): Merges a list of samples to form a mini-batch.
        batch_size (int, optional): The size of each batch. Defaults to 8.
        epochs (int, optional): The number of epochs for evaluation. Defaults to 1.

    Returns:
        tuple[int, dict[str, int]]: The overall loss and the losses by each head.
    """
    ds = ds.with_format(type="torch")
    loader = DataLoader(ds, batch_size=batch_size, collate_fn=collator)
    losses_by_head = defaultdict(list)
    losses = []
    for i, batch in tqdm(
        enumerate(loader), total=len(loader) * epochs, desc="Evaluating"
    ):
        outputs: HeadedModelOutput = model(**batch)
        for key in outputs.loss_by_head:
            losses_by_head[key].append(float(outputs.loss_by_head[key].item()))
        losses.append(float(outputs.loss.item()))
        if i >= len(loader) * epochs:
            break
    losses = float(np.mean(losses))
    losses_by_head = {
        key: float(np.mean(losses_by_head[key])) for key in losses_by_head
    }
    return losses, losses_by_head


@torch.inference_mode()
def get_some_preds(
    model,
    ds,
    tokenizer,
    n=5,
    classification=True,
) -> tuple[list[str], dict[str, list[int]], dict[str, list[int]]]:
    """
    Get predictions from the model.

    Args:
        model (HeadedModel): The model to be used for prediction.
        ds (Dataset): The dataset to be used for prediction.
        tokenizer (PreTrainedTokenizer): The tokenizer to be used.
        n (int, optional): The number of predictions to get (From the beginning of the datset). Defaults to 5.
        classification (bool, optional): Whether the task is text classification. Defaults to True.

    Returns:
        tuple[list[str], dict[str, list[int]], dict[str, list[int]]]: The inputs, predictions, and ground truths.
    """
    ds = ds.with_format(type="torch")
    loader = DataLoader(ds, batch_size=1)
    preds = defaultdict(list)
    inputs = []
    ground_truths = defaultdict(list)
    for i, batch in tqdm(
        enumerate(loader), total=min(n, len(loader)), desc="Predicting"
    ):
        inputs.append(tokenizer.decode(batch["input_ids"].squeeze()))
        outputs = model(**batch)
        for key in outputs.preds_by_head:
            if classification:
                p = outputs.preds_by_head[key][0, -1, :]
                p = torch.argmax(p).item()
                ground_truths[key].append(int(batch[key][0].item()))
            else:
                ground_truths[key].append(batch[key])
            preds[key].append(p)
        if i >= n:
            break
    return inputs, preds, ground_truths


@torch.inference_mode()
def get_top_n_preds(
    n: int,
    model: HeadedModel,
    text: str,
    tokenizer: PreTrainedTokenizer,
):
    """
    Get the top n predictions for a given text. Use for models with causal language modeling heads.

    Args:
        n (int): The number of top predictions to get.
        model (HeadedModel): The model to be used for prediction.
        text (str): The input text to be used for prediction.
        tokenizer (PreTrainedTokenizer): The tokenizer to be used.

    Returns:
        dict[str, list[str]]: The top n predictions for each head.
    """
    input = tokenizer(text, return_tensors="pt")
    output = model(**input)
    out = {}
    for head_name in output.preds_by_head:
        logits = output.preds_by_head[head_name]
        pred_logits = logits[0, -1, :]
        best_n = torch.topk(pred_logits, n)
        out[head_name] = [tokenizer.decode(i) for i in best_n.indices]
    return out
