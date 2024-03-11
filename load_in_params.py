from headed_model import get_multi_head_transformer
from transformers import PreTrainedModel
from typing import Type


def get_from_pretrained(base_model_class: Type[PreTrainedModel], model_name: str):
    model = get_multi_head_transformer(base_model_class)
