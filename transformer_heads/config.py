"""
This module defines the configuration classes for the model and its heads.

It includes the `HeadConfig` class, which defines the configuration for a model head, and the `HeadedConfig` class, 
which extends a base configuration class with additional output heads.

Classes:
    HeadConfig: A configuration class for a model head.
    HeadedConfig: A configuration class that extends a base configuration class with additional output heads.
"""

from dataclasses import asdict, dataclass
from typing import Optional, Type

from transformers import PretrainedConfig


@dataclass
class HeadConfig(dict):
    """
    A configuration class for a model head.

    Attributes:
        name (str): The name of the head.
        in_size (int): The input size for the head.
        num_outputs (Optional[int]): The number of outputs for the head.
        layer_hook (int): The layer to hook the head to. This uses python list indexing, so -1 is the last layer. Default is -1.
        hidden_size (int): The size of the hidden layers if the head should be an mlp. Default is 0.
        num_layers (int): The number of layers in the head. Set to 1 for a linear head and to > 1 for an mlp head. Default is 1.
        output_activation (str): The activation function for the output layer. Default is "linear".
        target (str): The name of the label column for this head. Defaults to name.
        is_causal_lm (Optional[bool]): Whether the head is for doing causal language modelling. Default is False.
        pred_for_sequence (Optional[bool]): Whether the head predicts on output per sequence (E.g text classification). Default is False.
        is_regression (Optional[bool]): Whether the head is for a regression task. Default is False.
        output_bias (Optional[bool]): Whether to include a bias in the output layer. Default is False.
        loss_fct (Optional[str]): The loss function for the head. Options are "cross_entropy", "mse", "bce". Default is "cross_entropy".
        trainable (Optional[bool]): Whether the head is trainable. Default is True.
        loss_weight (Optional[float]): The weight of this head when computing the loss. Default is 1.0.
        ignore_pads (Optional[bool]): Whether to ignore padding tokens when computing the loss. Default is False for causal_lm and True otherwise.
    """

    name: str
    in_size: int
    num_outputs: Optional[int]
    layer_hook: int = -1
    hidden_size: int = 0
    num_layers: int = 1
    output_activation: str = "linear"
    target: Optional[str] = None
    is_causal_lm: Optional[bool] = False
    pred_for_sequence: Optional[bool] = False
    is_regression: Optional[bool] = False
    output_bias: Optional[bool] = False
    loss_fct: Optional[str] = "cross_entropy"
    trainable: Optional[bool] = True
    loss_weight: Optional[float] = 1.0
    ignore_pads: Optional[bool] = None
    block_gradients: Optional[bool] = False

    def __hash__(self):
        return hash(tuple(sorted(self.items())))

    # Make minimally serializable
    # Caveat: Won't work without indent specified in json.dumps
    def items(self):
        return asdict(self).items()

    def __len__(self):
        return len(asdict(self))

    def __post_init__(self):
        if self.ignore_pads is None:
            self.ignore_pads = not self.is_causal_lm
        if self.target is None:
            self.target = self.name
        assert not (
            self.pred_for_sequence and self.is_causal_lm
        ), "Head cannot be both causal lm and predict for sequence"


def create_headed_model_config(
    base_config_class: Type[PretrainedConfig],
) -> Type[PretrainedConfig]:
    """
    Creates a new configuration class with additional output heads.

    This function takes a base configuration class and returns a new class that inherits from the base class
    and adds an `output_heads` attribute. The `output_heads` attribute is a list of `HeadConfig` instances.

    Args:
        base_config_class (Type[PretrainedConfig]): The base configuration class to extend.

    Returns:
        Type[PretrainedConfig]: A new configuration class that includes output heads.
    """

    class HeadedConfig(base_config_class):
        """
        A configuration class that extends a base configuration class with additional output heads.

        This class inherits from a base configuration class and adds an `output_heads` attribute.
        The `output_heads` attribute is a list of `HeadConfig` instances.

        Args:
            output_heads (List[Union[HeadConfig, dict]], optional): A list of `HeadConfig` instances or dictionaries
                that can be converted to `HeadConfig` instances. If not provided, an empty list is used.
            *args: Variable length argument list to be passed to the base configuration class.
            **kwargs: Arbitrary keyword arguments to be passed to the base configuration class.
        """

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
        def from_base_class(cls, base_config, output_heads) -> "HeadedConfig":
            """
            Creates a new instance of the class from a base configuration and a list of output heads.

            Args:
                base_config (PretrainedConfig): The base configuration to extend.
                output_heads (List[Union[HeadConfig, dict]]): A list of `HeadConfig` instances or dictionaries
                    that can be converted to `HeadConfig` instances.

            Returns:
                HeadedConfig: A new instance of the class with the provided base configuration and output heads.
            """
            out = cls(output_heads)
            for key, value in base_config.__dict__.items():
                setattr(out, key, value)
            return out

        def to_base_class(self) -> PretrainedConfig:
            """
            Converts the instance to a base configuration.

            This method creates a new instance of the base configuration class and copies all attributes from the current
            instance to the new base configuration instance.

            Returns:
                PretrainedConfig: A new instance of the base configuration class with the same attributes as the current instance.
            """
            base_cfg = base_config_class()
            for key in base_cfg.__dict__:
                setattr(base_cfg, key, self.__dict__[key])
            return base_cfg

    return HeadedConfig
