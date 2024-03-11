from transformers import PretrainedConfig
from transformers.models.mistral.modeling_mistral import MistralConfig
from dataclasses import dataclass
from typing import Optional, Type


@dataclass
class HeadConfig:
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


def create_headed_model_config(base_config_class: Type[PretrainedConfig]):
    class HeadedConfig(base_config_class):
        def __init__(self, output_heads, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            self.output_heads = (
                [HeadConfig(**head) for head in output_heads]
                if output_heads is not None
                else []
            )
            super().__init__(*args, **kwargs)

        def to_base_class(self):
            return base_config_class(*self.args, **self.kwargs)

    return HeadedConfig
