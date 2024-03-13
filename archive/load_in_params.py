from headed_model import get_multi_head_transformer
from transformers import PreTrainedModel
from transformers.utils import (
    cached_file,
    SAFE_WEIGHTS_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
)
from transformers.modeling_utils import load_state_dict
from transformers.utils.hub import get_checkpoint_shard_files
from typing import Type


def get_cached_weights(model_name):
    resolved_archive_file = cached_file(model_name, SAFE_WEIGHTS_NAME)
    if resolved_archive_file is None:
        resolved_archive_file = cached_file(
            model_name,
            SAFE_WEIGHTS_INDEX_NAME,
        )
        resolved_archive_file, sharded_metadata = get_checkpoint_shard_files(
            model_name,
            resolved_archive_file,
        )
        loaded_state_dict_keys = sharded_metadata["all_checkpoint_keys"]
    else:
        state_dict = load_state_dict(resolved_archive_file)
        loaded_state_dict_keys = list(state_dict.keys())


def get_from_pretrained(base_model_class: Type[PreTrainedModel], model_name: str):
    config = base_model_class.config_class.from_pretrained(model_name)
    model = get_multi_head_transformer(base_model_class)
