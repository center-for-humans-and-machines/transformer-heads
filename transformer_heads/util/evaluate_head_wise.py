from transformer_heads.model.model import HeadedModel
from transformer_heads.output import HeadedModelOutput
from datasets import Dataset
from torch.utils.data import DataLoader
from collections import defaultdict
import torch
import numpy as np
from tqdm import tqdm


@torch.inference_mode()
def evaluate(model: HeadedModel, ds: Dataset, collator=None, batch_size=8):
    ds = ds.with_format(type="torch")
    loader = DataLoader(ds, batch_size=batch_size, collate_fn=collator)
    losses_by_head = defaultdict(list)
    losses = []
    for batch in tqdm(loader, desc="Evaluating"):
        outputs: HeadedModelOutput = model(**batch)
        for key in outputs.loss_by_head:
            losses_by_head[key].append(float(outputs.loss_by_head[key].item()))
        losses.append(float(outputs.loss.item()))
    losses = float(np.mean(losses))
    losses_by_head = {
        key: float(np.mean(losses_by_head[key])) for key in losses_by_head
    }
    return losses, losses_by_head
