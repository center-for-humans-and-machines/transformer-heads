from transformer_heads.model.model import HeadedModel
from transformer_heads.output import HeadedModelOutput
from transformers import PreTrainedTokenizer
from datasets import Dataset
from torch.utils.data import DataLoader
from collections import defaultdict
import torch
import numpy as np
from tqdm import tqdm


@torch.inference_mode()
def evaluate_head_wise(
    model: HeadedModel, ds: Dataset, collator=None, batch_size=8, epochs=1
):
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
def get_top_n_preds(
    n: int,
    model: HeadedModel,
    text: str,
    tokenizer: PreTrainedTokenizer,
):
    input = tokenizer(text, return_tensors="pt")
    output = model(**input)
    out = {}
    for head_name in output.logits_by_head:
        logits = output.logits_by_head[head_name]
        pred_logits = logits[0, -1, :]
        best_n = torch.topk(pred_logits, n)
        out[head_name] = [tokenizer.decode(i) for i in best_n.indices]
    return out
