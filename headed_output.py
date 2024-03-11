from transformers.utils import ModelOutput
from dataclasses import dataclass
import torch
from typing import Optional, Tuple


@dataclass
class HeadedModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor]
    loss_by_head: Optional[dict[str, torch.FloatTensor]]
    logits_by_head: Optional[dict[str, torch.FloatTensor]]
    preds_by_head: Optional[dict[str, torch.FloatTensor]]
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
