import torch.nn as nn
from transformers import MistralModel

loss_fct_map = {
    "mse": nn.MSELoss(),
    "cross_entropy": nn.CrossEntropyLoss(),
}

model_type_map = {
    "mistral": MistralModel,
}
