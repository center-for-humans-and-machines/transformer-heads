import torch.nn as nn
from transformers import (
    MistralModel,
    GPT2Model,
    LlamaModel,
)

loss_fct_map = {
    "mse": nn.MSELoss(),
    "cross_entropy": nn.CrossEntropyLoss(),
    "bce": nn.BCELoss(),
}

model_type_map = {
    "mistral": ("model", MistralModel),
    "gpt2": ("transformer", GPT2Model),
    "llama": ("model", LlamaModel),
}
