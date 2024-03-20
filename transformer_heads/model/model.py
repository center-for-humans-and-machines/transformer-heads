from abc import ABC
from os import PathLike
from typing import Any, Callable, Dict, List, Optional, Type

import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast

from transformer_heads.config import HeadConfig, create_headed_model_config
from transformer_heads.constants import loss_fct_map, model_type_map
from transformer_heads.model.head import MLPHead
from transformer_heads.output import HeadedModelOutput


def get_headed_pretrained_model_class(base_model_class: Type[PreTrainedModel]):
    class HeadedPreTrainedModel(base_model_class):
        config_class = create_headed_model_config(base_model_class.config_class)

    return HeadedPreTrainedModel


class HeadedModel(ABC, PreTrainedModel):
    head_configs: List[HeadConfig]
    vocab_size: int
    heads: nn.ModuleDict
    lm_head_config: Optional[HeadConfig]
    lm_head: Optional[MLPHead]


def get_multi_head_transformer(base_model_class: Type[PreTrainedModel]):
    class TransformerWithHeads(
        get_headed_pretrained_model_class(base_model_class), HeadedModel
    ):
        def __init__(self, config: PretrainedConfig):
            super().__init__(config)
            setattr(
                self,
                model_type_map[config.model_type][0],
                model_type_map[config.model_type][1](config.to_base_class()),
            )
            self.vocab_size: int = config.vocab_size
            self.head_configs: dict[str:HeadConfig] = {
                cfg.name: cfg for cfg in config.output_heads
            }
            self.heads = nn.ModuleDict(
                {
                    name: MLPHead.from_head_config(head_config)
                    for name, head_config in self.head_configs.items()
                }
            )

            # Make pretrained loading of lm_head work
            self.lm_head = None
            self.lm_head_config = None
            head: MLPHead
            for name, head in self.heads.items():
                if name == "lm_head":
                    self.lm_head = head.lins[0]
                    self.lm_head_config = self.head_configs[name]
                    del self.heads[name]
                    break
            self._hf_peft_config_loaded = False

        def save_pretrained(
            self,
            save_directory: str | PathLike,
            is_main_process: bool = True,
            state_dict: Dict | None = None,
            save_function: Callable[..., Any] = torch.save,
            push_to_hub: bool = False,
            max_shard_size: int | str = "5GB",
            safe_serialization: bool = True,
            variant: str | None = None,
            token: str | bool | None = None,
            save_peft_format: bool = True,
            **kwargs,
        ):
            super().save_pretrained(
                save_directory,
                is_main_process,
                state_dict,
                save_function,
                push_to_hub,
                max_shard_size,
                safe_serialization,
                variant,
                token,
                save_peft_format,
                **kwargs,
            )
            head: MLPHead
            for head in self.heads.values():
                if head.requires_individual_saving:
                    head.save_to_safetensors(save_directory)

        def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = False,
            **labels,
        ) -> HeadedModelOutput:
            assert not return_dict
            output_attentions = (
                output_attentions
                if output_attentions is not None
                else self.config.output_attentions
            )
            output_hidden_states = (
                output_hidden_states
                if output_hidden_states is not None
                else self.config.output_hidden_states
            )

            # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
            outputs: BaseModelOutputWithPast = getattr(
                self, model_type_map[self.config.model_type][0], "model"
            )(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=True,
            )

            out_preds = {}

            hidden_states = outputs.hidden_states
            loss = torch.tensor(
                0.0, device=input_ids.device, dtype=torch.float32, requires_grad=True
            )
            loss_by_head = {}
            for key in list(self.heads.keys()) + ["lm_head"]:
                if key == "lm_head":
                    if self.lm_head is None:
                        continue
                    head = self.lm_head
                    head_config = self.lm_head_config
                else:
                    head = self.heads[key]
                    head_config = self.head_configs[key]
                selected_hidden_states = hidden_states[head_config.layer_hook]
                logits: torch.FloatTensor = head(selected_hidden_states)
                out_preds[head_config.name] = logits
                if (
                    labels is not None
                    and head_config.name in labels
                    and head_config.loss_fct is not None
                ):
                    loss_fct = loss_fct_map[head_config.loss_fct]
                    if head_config.is_causal_lm:
                        use_logits = logits[..., :-1, :].contiguous()
                        use_labels = labels[head_config.name][..., 1:].contiguous()
                    else:
                        use_logits = logits
                        use_labels = labels[head_config.name]
                    if head_config.pred_for_sequence:
                        use_logits = use_logits[..., -1, :].contiguous()
                    if head_config.is_regression:
                        use_logits = use_logits.view(-1)
                    else:
                        use_logits = use_logits.view(
                            -1, head_config.num_outputs or self.config.vocab_size
                        )
                    use_labels = use_labels.view(-1)
                    use_labels = use_labels.to(use_logits.device)
                    loss_by_head[head_config.name] = loss_fct(use_logits, use_labels)
                    loss = (
                        loss + loss_by_head[head_config.name] * head_config.loss_weight
                    )

            return HeadedModelOutput(
                loss=loss,
                loss_by_head=loss_by_head,
                preds_by_head=out_preds,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states if output_hidden_states else None,
                attentions=outputs.attentions,
            )

    return TransformerWithHeads
