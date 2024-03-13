import os

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file

from transformer_heads.config import HeadConfig
from transformer_heads.util.model import patch_state_dict

activation_map = {"sigmoid": nn.Sigmoid, "linear": nn.Identity, "relu": nn.ReLU}


class MLPHead(nn.Module):
    @classmethod
    def from_head_config(cls, head_config: HeadConfig):
        return cls(
            head_config.name,
            head_config.in_size,
            head_config.hidden_size,
            head_config.num_layers,
            head_config.output_activation,
            head_config.num_outputs or 1,
            head_config.output_bias,
        )

    def __init__(
        self,
        name: str,
        in_size,
        hidden_size,
        num_layers,
        output_activation: str,
        num_outputs: int = 1,
        output_bias: bool = False,
    ):
        super().__init__()
        self.name = name
        self.lins = nn.ModuleList()
        if num_layers == 1:
            self.lins.append(nn.Linear(in_size, num_outputs, bias=output_bias))
        else:
            self.lins.append(nn.Linear(in_size, hidden_size, bias=True))
            for _ in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_size, hidden_size, bias=True))
            self.lins.append(nn.Linear(hidden_size, num_outputs, bias=output_bias))

        self.hidden_activation = nn.ReLU()
        self.output_activation = activation_map[output_activation]()
        self.requires_individual_saving = False

    def set_requires_grad(self, requires_grad):
        for name, param in self.named_parameters():
            param.requires_grad = requires_grad

    def save_to_safetensors(self, folder):
        save_file(self.state_dict(), os.path.join(folder, self.name + ".safetensors"))

    def load_from_safetensors(self, folder):
        self.load_state_dict(
            patch_state_dict(
                load_file(os.path.join(folder, self.name + ".safetensors"))
            )
        )

    def forward(self, x) -> torch.FloatTensor:
        for i, lin in enumerate(self.lins):
            x = lin(x)
            if i < len(self.lins) - 1:
                x = self.hidden_activation(x)
        x = self.output_activation(x)
        return x
