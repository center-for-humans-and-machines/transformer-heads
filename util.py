import torch


def patch_state_dict(state_dict):
    return {
        key: value if value.dim() > 0 else torch.unsqueeze(value, 0)
        for key, value in state_dict.items()
    }
