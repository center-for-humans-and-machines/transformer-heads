from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from transformers.utils import ModelOutput


@dataclass
class HeadedModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    loss_by_head: Optional[dict[str, torch.FloatTensor]] = None
    preds_by_head: Optional[dict[str, torch.FloatTensor]] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
