from transformers.models.mistral.modeling_mistral import (
    MistralModel,
    MistralPreTrainedModel,
)
from transformers.modeling_outputs import BaseModelOutputWithPast
from headed_mistral_config import HeadConfig
from headed_output import HeadedModelOutput
from mlp_head import MLPHead
import torch.nn as nn
import torch
from typing import Optional, List, Union, Tuple, Dict, Type

from transformers import PreTrainedModel, PretrainedConfig
from headed_mistral_config import create_headed_model_config


def get_headed_pretrained_model_class(base_model_class: Type[PreTrainedModel]):
    class HeadedPreTrainedModel(base_model_class):
        config_class = create_headed_model_config(base_model_class.config_class)

    return HeadedPreTrainedModel


loss_fct_map = {
    "mse": nn.MSELoss(),
    "cross_entropy": nn.CrossEntropyLoss(),
}

model_type_map = {
    "mistral": MistralModel,
}


def get_multi_head_transformer(base_model_class: Type[PreTrainedModel]):
    class TransformerWithHeads(get_headed_pretrained_model_class(base_model_class)):
        def __init__(self, config: PretrainedConfig):
            super().__init__(config)
            self.model = model_type_map[config.model_type](config.to_base_class())
            self.vocab_size: int = config.vocab_size
            self.head_configs: List[HeadConfig] = config.output_heads
            self.heads = nn.ModuleList(
                [
                    MLPHead.from_head_config(head_config)
                    for head_config in config.output_heads
                ]
            )
            self.lm_head = None
            head: MLPHead
            for head_config, head in zip(self.head_configs, self.heads):
                if head_config.name == "lm_head":
                    self.lm_head = head.lins[0]
                    break

        def get_input_embeddings(self):
            return self.model.embed_tokens

        def set_input_embeddings(self, value):
            self.model.embed_tokens = value

        def set_decoder(self, decoder):
            self.model = decoder

        def get_decoder(self):
            return self.model

        def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[
                Dict[str, Union[torch.LongTensor, torch.FloatTensor]]
            ] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
        ) -> HeadedModelOutput:
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
            return_dict = (
                return_dict if return_dict is not None else self.config.use_return_dict
            )

            # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
            outputs: BaseModelOutputWithPast = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=True,
                return_dict=return_dict,
            )

            out_logits = {}
            out_preds = {}

            hidden_states = outputs.hidden_states
            loss = 0
            loss_by_head = {}
            for head, head_config in (self.heads, self.head_configs):
                selected_hidden_states = hidden_states[head_config.layer_hook]
                logits: torch.FloatTensor = head(selected_hidden_states)
                if head_config.is_regression:
                    out_preds[head_config.name] = logits
                else:
                    out_logits[head_config.name] = logits
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
                    if head_config.is_regression:
                        use_logits = use_logits.view(-1)
                    else:
                        use_logits = use_logits.view(
                            -1, head_config.num_outputs or self.config.vocab_size
                        )
                    use_labels = use_labels.view(-1)
                    use_labels = use_labels.to(use_logits.device)
                    loss_by_head[head_config.name] = loss_fct(
                        use_logits, use_labels[head_config.name]
                    )
                    loss += loss_by_head[head_config.name]

            return HeadedModelOutput(
                loss=loss,
                loss_by_head=loss_by_head,
                logits_by_head=out_logits,
                preds_by_head=out_preds,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states if output_hidden_states else None,
                attentions=outputs.attentions,
            )

    return TransformerWithHeads
