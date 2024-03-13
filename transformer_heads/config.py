from dataclasses import asdict, dataclass
from typing import Optional, Type

from transformers import PretrainedConfig


@dataclass
class HeadConfig(dict):
    name: str
    layer_hook: int
    in_size: int
    hidden_size: int
    num_layers: int
    output_activation: str
    is_causal_lm: Optional[bool]
    loss_fct: Optional[str]
    num_outputs: Optional[int]
    is_regression: Optional[bool]
    output_bias: Optional[bool] = False

    def __hash__(self):
        return hash(tuple(sorted(self.items())))

    # Make minimally serializable
    # Caveat: Won't work without indent specified in json.dumps
    def items(self):
        return asdict(self).items()

    def __len__(self):
        return len(asdict(self))


def create_headed_model_config(base_config_class: Type[PretrainedConfig]):
    class HeadedConfig(base_config_class):
        def __init__(self, output_heads=None, *args, **kwargs):
            self.output_heads = (
                [
                    (head if isinstance(head, HeadConfig) else HeadConfig(**head))
                    for head in output_heads
                ]
                if output_heads is not None
                else []
            )
            super().__init__(*args, **kwargs)

        @classmethod
        def from_base_class(cls, base_config, output_heads):
            out = cls(output_heads)
            for key, value in base_config.__dict__.items():
                setattr(out, key, value)
            return out

        def to_base_class(self):
            base_cfg = base_config_class()
            for key in base_cfg.__dict__:
                setattr(base_cfg, key, self.__dict__[key])
            return base_cfg

    return HeadedConfig
