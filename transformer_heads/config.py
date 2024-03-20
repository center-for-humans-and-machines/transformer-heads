from dataclasses import asdict, dataclass
from typing import Optional, Type

from transformers import PretrainedConfig


@dataclass
class HeadConfig(dict):
    name: str
    in_size: int
    num_outputs: Optional[int]
    layer_hook: int = -1
    hidden_size: int = 0
    num_layers: int = 1
    output_activation: str = "linear"
    is_causal_lm: Optional[bool] = False
    pred_for_sequence: Optional[bool] = False
    is_regression: Optional[bool] = False
    output_bias: Optional[bool] = False
    loss_fct: Optional[str] = "cross_entropy"
    trainable: Optional[bool] = True
    loss_weight: Optional[float] = 1.0

    def __hash__(self):
        return hash(tuple(sorted(self.items())))

    # Make minimally serializable
    # Caveat: Won't work without indent specified in json.dumps
    def items(self):
        return asdict(self).items()

    def __len__(self):
        return len(asdict(self))

    def __post_init__(self):
        assert not (self.pred_for_sequence and self.is_causal_lm)


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
