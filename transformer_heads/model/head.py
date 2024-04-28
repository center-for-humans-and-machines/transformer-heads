"""
This module defines a class `MLPHead` that represents a multi-layer perceptron (MLP) head for a transformer model. 
It also includes utility functions for saving and loading the state of the MLP head.

Classes:
    MLPHead: A class that represents a multi-layer perceptron (MLP) head for a transformer model.
"""

import os

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file

from transformer_heads.config import HeadConfig
from transformer_heads.util.model import patch_state_dict
from transformer_heads.constants import activation_map


class MLPHead(nn.Module):
    """
    A class that represents a multi-layer perceptron (MLP) head for a transformer model.

    Attributes:
        name (str): The name of the MLP head.
        trainable (bool): Whether the MLP head is trainable.
        lins (nn.ModuleList): A list of linear layers in the MLP head.
        hidden_activation (nn.ReLU): The activation function for the hidden layers.
        output_activation (nn.Module): The activation function for the output layer.
        requires_individual_saving (bool): Whether the MLP head needs to be saved separately.
    """

    def __init__(
        self,
        name: str,
        in_size,
        hidden_size,
        num_layers,
        output_activation: str,
        num_outputs: int = 1,
        output_bias: bool = False,
        trainable: bool = True,
        block_gradients: bool = False,
    ):
        super().__init__()
        self.name = name
        self.trainable = trainable
        self.block_gradients = block_gradients
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

    @classmethod
    def from_head_config(cls, head_config: HeadConfig) -> "MLPHead":
        """
        Creates an MLP head from a head configuration.

        Args:
            head_config (HeadConfig): The head configuration.

        Returns:
            MLPHead: The created MLP head.
        """
        return cls(
            head_config.name,
            head_config.in_size,
            head_config.hidden_size,
            head_config.num_layers,
            head_config.output_activation,
            head_config.num_outputs or 1,
            head_config.output_bias,
            head_config.trainable,
            head_config.block_gradients,
        )

    def set_requires_grad(self, requires_grad):
        """
        Sets whether the parameters of the MLP head require gradients.

        Args:
            requires_grad (bool): Whether the parameters require gradients.
        """
        assert not requires_grad or self.trainable
        for _name, param in self.named_parameters():
            param.requires_grad = requires_grad

    def save_to_safetensors(self, folder):
        """
        Saves the state of the MLP head to a safetensors file.

        Args:
            folder (str): The folder where the file will be saved.
        """
        save_file(self.state_dict(), os.path.join(folder, self.name + ".safetensors"))

    def load_from_safetensors(self, folder):
        """
        Loads the state of the MLP head from a safetensors file.

        Args:
            folder (str): The folder where the file is located.
        """
        self.load_state_dict(
            patch_state_dict(
                load_file(os.path.join(folder, self.name + ".safetensors"))
            )
        )

    def forward(self, x) -> torch.FloatTensor:
        """
        Performs a forward pass through the MLP head.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.FloatTensor: The output tensor.
        """
        if self.block_gradients:
            x = x.detach()
        for i, lin in enumerate(self.lins):
            x = lin(x)
            if i < len(self.lins) - 1:
                x = self.hidden_activation(x)
        x = self.output_activation(x)
        return x
