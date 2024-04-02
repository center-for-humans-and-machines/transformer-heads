import torch


class Masked_MSE_Loss(torch.nn.MSELoss):
    def forward(self, input: torch.Tensor, target: torch.Tensor):
        mask = target != -100
        return super().forward(input[mask], target[mask])
