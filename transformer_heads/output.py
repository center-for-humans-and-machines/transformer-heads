"""
This module defines the output class for a model with multiple heads.

It includes the `HeadedModelOutput` class, which extends the `ModelOutput` class from the transformers library 
with additional attributes for multi-head models.

Classes:
    HeadedModelOutput: An output class for a model with multiple heads.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from transformers.utils import ModelOutput


@dataclass
class HeadedModelOutput(ModelOutput):
    """
    An output class for a model with multiple heads.

    This class extends the `ModelOutput` class from the transformers library with additional attributes for multi-head models.

    Attributes:
        loss (Optional[torch.FloatTensor]): The total loss.
        loss_by_head (Optional[dict[str, torch.FloatTensor]]): A dictionary mapping head names to their corresponding losses.
        preds_by_head (Optional[dict[str, torch.FloatTensor]]): A dictionary mapping head names to their corresponding predictions.
        past_key_values (Optional[Tuple[Tuple[torch.FloatTensor]]]): Tuple of key value states for transformer models.
        hidden_states (Optional[Tuple[torch.FloatTensor, ...]]): Tuple of hidden states for transformer models.
        attentions (Optional[Tuple[torch.FloatTensor, ...]]): Tuple of attention weights for transformer models.
    """

    loss: Optional[torch.FloatTensor] = None
    loss_by_head: Optional[dict[str, torch.FloatTensor]] = None
    adapted_loss_by_head: Optional[dict[str, torch.FloatTensor]] = None
    preds_by_head: Optional[dict[str, torch.FloatTensor]] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class HeadedModelGenerateOutput(ModelOutput):
    """
    An generation output class for a model with multiple heads.
    """

    sequences: torch.LongTensor = None
    logprobs: torch.FloatTensor = None
    head_outputs: dict[str, torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
