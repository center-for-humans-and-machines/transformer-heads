"""
This module provides classes and functions for creating and manipulating transformer models with multiple heads.

Classes:
    HeadedModel: Abstract base class for models with multiple heads.
    TransformerWithHeads: Transformer model with multiple heads.

Functions:
    get_headed_pretrained_model_class: Get a new model base class for a pretrained model with multiple heads.
    get_multi_head_transformer: Patch a pretrained transformer model to add multiple heads.
"""

import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from os import PathLike
from typing import Any, Callable, Dict, List, Optional, Type, Union

import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import log_softmax
from transformers import GenerationConfig, PretrainedConfig, PreTrainedModel
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import (
    EosTokenCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)
from transformers.generation.streamers import BaseStreamer
from transformers.generation.utils import NEED_SETUP_CACHE_CLASSES_MAPPING, logger
from transformers.modeling_outputs import BaseModelOutputWithPast

from transformer_heads.config import HeadConfig, create_headed_model_config
from transformer_heads.constants import loss_fct_map, model_type_map
from transformer_heads.model.head import MLPHead
from transformer_heads.output import HeadedModelGenerateOutput, HeadedModelOutput
from transformer_heads.util.helpers import Welfords


def get_headed_pretrained_model_class(base_model_class: Type[PreTrainedModel]):
    """
    Get a new model base class for a pretrained model with multiple heads.

    Args:
        base_model_class (Type[PreTrainedModel]): The base class of the pretrained model.

    Returns:
        Type[PreTrainedModel]: The new class supporting headed model configuration.
    """

    class HeadedPreTrainedModel(base_model_class):
        config_class = create_headed_model_config(base_model_class.config_class)

    return HeadedPreTrainedModel


class HeadedModel(ABC, PreTrainedModel):
    """
    Abstract base class for models with multiple heads.

    Attributes:
        head_configs (List[HeadConfig]): The configurations for the new heads.
        vocab_size (int): The size of the vocabulary.
        heads (nn.ModuleDict): The new heads of the model.
        lm_head_config (Optional[HeadConfig]): The configuration for the pretrained language model head.
        lm_head (Optional[MLPHead]): The pretrained language model head.
    """

    head_configs: dict[str, HeadConfig]
    vocab_size: int
    heads: nn.ModuleDict
    lm_head_config: Optional[HeadConfig]
    lm_head: Optional[nn.Module]
    adaptive_loss: bool
    adaptive_warmup: Optional[int]
    adaptive_collect: Optional[dict[str, Welfords]]

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
        """
        Forward pass of the model.

        Args:
            input_ids (torch.LongTensor, optional): The input IDs. Defaults to None.
            attention_mask (Optional[torch.Tensor], optional): The attention mask. Defaults to None.
            position_ids (Optional[torch.LongTensor], optional): The position IDs. Defaults to None.
            past_key_values (Optional[List[torch.FloatTensor]], optional): The past key values. Defaults to None.
            inputs_embeds (Optional[torch.FloatTensor], optional): The input embeddings. Defaults to None.
            use_cache (Optional[bool], optional): Whether to use cache. Defaults to None.
            output_attentions (Optional[bool], optional): Whether to output attentions. Defaults to None.
            output_hidden_states (Optional[bool], optional): Whether to output hidden states. Defaults to None.
            return_dict (Optional[bool], optional): Not supported, keep as False. Defaults to False.
            **labels: The labels for the heads.

        Returns:
            HeadedModelOutput: The output of the model.
        """
        assert not return_dict, "return_dict not supported for transformer_heads models"
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

        if any([head_config.ignore_pads for head_config in self.head_configs.values()]):
            pad_tk_id = (
                self.config.pad_token_id
                if self.config.pad_token_id is not None
                else self.config.eos_token_id
            )
            assert (
                pad_tk_id is not None
            ), "Model must have pad token id set if any head ignores pads."
            if isinstance(pad_tk_id, list):
                sequence_lengths = (
                    torch.isin(input_ids, torch.tensor(pad_tk_id).to(input_ids.device))
                    .int()
                    .argmax(-1)
                    - 1
                )
            else:
                sequence_lengths = torch.eq(input_ids, pad_tk_id).int().argmax(-1) - 1
            sequence_lengths = sequence_lengths % input_ids.shape[-1]
            sequence_lengths = sequence_lengths.to(outputs.hidden_states[0].device)

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
                and head_config.target in labels
                and head_config.loss_fct is not None
                and labels[head_config.target] is not None
            ):
                loss_fct = loss_fct_map[head_config.loss_fct]
                if head_config.is_causal_lm:
                    use_logits = logits[..., :-1, :].contiguous()
                    use_labels = labels[head_config.target][..., 1:].contiguous()
                else:
                    use_logits = logits
                    use_labels = labels[head_config.target]
                if head_config.pred_for_sequence:
                    if head_config.ignore_pads:
                        use_logits = logits[
                            torch.arange(logits.shape[0], device=logits.device),
                            sequence_lengths,
                        ]
                    else:
                        use_logits = use_logits[..., -1, :].contiguous()
                if head_config.ignore_pads and not head_config.pred_for_sequence:
                    use_logits = torch.concatenate(
                        [
                            use_logits[i, : sequence_lengths[i]]
                            for i in range(use_logits.shape[0])
                        ]
                    )
                    use_labels = torch.concatenate(
                        [
                            use_labels[i, : sequence_lengths[i]]
                            for i in range(use_labels.shape[0])
                        ]
                    )
                else:
                    use_labels = use_labels.view(-1)

                if head_config.is_regression:
                    use_logits = use_logits.view(-1)
                else:
                    use_logits = use_logits.view(
                        -1, head_config.num_outputs or self.config.vocab_size
                    )
                use_labels = use_labels.to(use_logits.device)
                loss_by_head[head_config.name] = loss_fct(use_logits, use_labels)

                # Now let's prevent ddp errors for some None gradients
                # If cross-entropy loss and all are ignored, loss can be nan at this point
                if not torch.all(loss_by_head[head_config.name].isfinite()):
                    loss_by_head[head_config.name] = torch.sum(use_logits * 0.0)

        if self.adaptive_loss:
            adapted_losses = self.adapt_losses(loss_by_head)
        loss = sum(
            [
                value
                * (
                    self.lm_head_config.loss_weight
                    if key == "lm_head"
                    else self.head_configs[key].loss_weight
                )
                for key, value in (
                    adapted_losses if self.adaptive_loss else loss_by_head
                ).items()
                if torch.all(value.isfinite())
            ],
            loss,
        )
        if self.adaptive_loss:
            if self.training:
                for key, value in loss_by_head.items():
                    if torch.all(torch.isfinite(value)):
                        val = value.item()
                        if val != 0.0:
                            self.adaptive_collect[key].update(val)

        return HeadedModelOutput(
            loss=loss,
            loss_by_head=loss_by_head,
            adapted_loss_by_head=(
                adapted_losses if self.adaptive_loss else loss_by_head
            ),
            preds_by_head=out_preds,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
        )

    def set_adaptive_loss(self, enable: bool, warmup_steps=5):
        """
        Sets the adaptive loss for the model.

        Args:
            enable (bool): Whether to enable adaptive loss.
            warmup_steps (int, optional): The number of warmup steps. Defaults to 5.

        Raises:
            AssertionError: If adaptive loss is enabled and any head has a loss weight other than 1.0.
        """
        self.adaptive_loss = enable
        if self.adaptive_loss:
            self.adaptive_collect = defaultdict(Welfords)
            self.adaptive_warmup = warmup_steps

    def adapt_losses(self, loss_by_head):
        """
        Adapts the losses for each head in the model.

        If the number of losses in the history for each head is less than the number of warmup steps,
        the function returns a dictionary with the keys being the head names and the values being 0.
        Otherwise, the function calculates the new loss for each head by subtracting the mean of the
        loss history from the current loss and dividing by the standard deviation of the loss history.

        Args:
            loss_by_head (dict[str, float]): A dictionary with the keys being the head names and the values being the current losses.

        Returns:
            dict[str, float]: A dictionary with the keys being the head names and the values being the new losses.
        """
        if len(self.adaptive_collect) == 0:
            return {key: value * 0 for key, value in loss_by_head.items()}
        else:
            new_loss_by_head = {}
            for key, loss in loss_by_head.items():
                if self.adaptive_collect[key].count < self.adaptive_warmup:
                    new_loss_by_head[key] = loss * 0.0
                elif loss == 0.0:
                    new_loss_by_head[key] = loss
                else:
                    new_loss_by_head[key] = (
                        loss - self.adaptive_collect[key].mean
                    ) / np.clip(self.adaptive_collect[key].std, 1e-6, None)
            return new_loss_by_head

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        prefix_allowed_tokens_fn: Optional[
            Callable[[int, torch.Tensor], List[int]]
        ] = None,
        **kwargs,
    ) -> HeadedModelGenerateOutput:
        r"""

        Generates sequences of token ids for models with a language modeling head.

        Parameters:
            inputs (`torch.Tensor` of varying shape depending on the modality, *optional*):
                The sequence used as a prompt for the generation or as model inputs to the encoder. If `None` the
                method initializes it with `bos_token_id` and a batch size of 1. For decoder-only models `inputs`
                should be in the format of `input_ids`. For encoder-decoder models *inputs* can represent any of
                `input_ids`, `input_values`, `input_features`, or `pixel_values`.
            generation_config (`~generation.GenerationConfig`, *optional*):
                The generation configuration to be used as base parametrization for the generation call. `**kwargs`
                passed to generate matching the attributes of `generation_config` will override them. If
                `generation_config` is not provided, the default will be used, which has the following loading
                priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
                configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
                default values, whose documentation should be checked to parameterize generation.
            logits_processor (`LogitsProcessorList`, *optional*):
                Custom logits processors that complement the default logits processors built from arguments and
                generation config. If a logit processor is passed that is already created with the arguments or a
                generation config an error is thrown. This feature is intended for advanced users.
            prefix_allowed_tokens_fn (`Callable[[int, torch.Tensor], List[int]]`, *optional*):
                If provided, this function constraints the beam search to allowed tokens only at each step. If not
                provided no constraint is applied. This function takes 2 arguments: the batch ID `batch_id` and
                `input_ids`. It has to return a list with the allowed tokens for the next generation step conditioned
                on the batch ID `batch_id` and the previously generated tokens `inputs_ids`. This argument is useful
                for constrained generation conditioned on the prefix, as described in [Autoregressive Entity
                Retrieval](https://arxiv.org/abs/2010.00904).
            kwargs (`Dict[str, Any]`, *optional*):
                Ad hoc parametrization of `generation_config` and/or additional model-specific kwargs that will be
                forwarded to the `forward` function of the model. If the model is an encoder-decoder model, encoder
                specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder_*.

        Return:
            HeadedModelGenerateOutput: Contains the generated sequences, the log probabilities and the head outputs
        """
        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        generation_config, model_kwargs = self._prepare_generation_config(
            generation_config, **kwargs
        )
        self._validate_model_kwargs(model_kwargs.copy())

        if not hasattr(generation_config, "_eos_token_tensor"):
            setattr(
                generation_config,
                "_eos_token_tensor",
                torch.tensor(
                    [generation_config.eos_token_id],
                    device=self.device,
                    dtype=torch.long,
                ),
            )

        # 2. Set generation parameters if not already defined
        logits_processor = (
            logits_processor if logits_processor is not None else LogitsProcessorList()
        )

        if (
            generation_config.pad_token_id is None
            and generation_config.eos_token_id is not None
        ):
            eos_token_id = generation_config.eos_token_id
            if isinstance(eos_token_id, list):
                eos_token_id = eos_token_id[0]
            generation_config.pad_token_id = eos_token_id

        # 3. Define model inputs
        # inputs_tensor has to be defined
        # model_input_name is defined if model-specific keyword input is passed
        # otherwise model_input_name is None
        # all model-specific keyword inputs are removed from `model_kwargs`
        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )
        batch_size = inputs_tensor.shape[0]

        # 4. Define other model kwargs
        if not self.config.is_encoder_decoder and model_input_name == "inputs_embeds":
            model_kwargs["use_cache"] = True
        else:
            model_kwargs["use_cache"] = generation_config.use_cache

        accepts_attention_mask = True
        requires_attention_mask = "encoder_outputs" not in model_kwargs

        if (
            model_kwargs.get("attention_mask", None) is None
            and requires_attention_mask
            and accepts_attention_mask
        ):
            model_kwargs["attention_mask"] = (
                self._prepare_attention_mask_for_generation(
                    inputs_tensor,
                    torch.tensor(
                        [generation_config.pad_token_id],
                        device=inputs_tensor.device,
                        dtype=torch.long,
                    ),
                    generation_config._eos_token_tensor,
                )
            )

        # decoder-only models should use left-padding for generation
        if not self.config.is_encoder_decoder:
            # If `input_ids` was given, check if the last id in any sequence is `pad_token_id`
            # Note: If using, `inputs_embeds` this check does not work, because we want to be more hands-off.
            if (
                generation_config.pad_token_id is not None
                and len(inputs_tensor.shape) == 2
                and torch.sum(inputs_tensor[:, -1] == generation_config.pad_token_id)
                > 0
            ):
                logger.warning(
                    "A decoder-only architecture is being used, but right-padding was detected! For correct "
                    "generation results, please set `padding_side='left'` when initializing the tokenizer."
                )

        # 5. Prepare `input_ids` which will be used for auto-regressive generation
        input_ids = (
            inputs_tensor
            if model_input_name == "input_ids"
            else model_kwargs.pop("input_ids")
        )

        # 6. Prepare `max_length` depending on other stopping criteria.
        input_ids_length = input_ids.shape[-1]
        has_default_max_length = (
            kwargs.get("max_length") is None
            and generation_config.max_length is not None
        )
        has_default_min_length = (
            kwargs.get("min_length") is None
            and generation_config.min_length is not None
        )
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            has_default_min_length=has_default_min_length,
            model_input_name=model_input_name,
            inputs_tensor=inputs_tensor,
            input_ids_length=input_ids_length,
        )

        if generation_config.cache_implementation in NEED_SETUP_CACHE_CLASSES_MAPPING:
            if generation_config.cache_implementation == "static":
                if model_kwargs.get("past_key_values", False) is not False:
                    raise ValueError(
                        "Using `past_key_values` argument with `generate()` when using a static KV cache is not supported. Please open an issue in Transformers GitHub repository."
                    )
                cache_cls = NEED_SETUP_CACHE_CLASSES_MAPPING["static"]
                if not callable(getattr(self, "_setup_cache", None)):
                    raise ValueError(
                        "The `generation_config` defines a `cache_implementation` that is not compatible with this model."
                        " Make sure it has a `_setup_cache` function."
                    )
                self._setup_cache(
                    cache_cls,
                    max_batch_size=batch_size,
                    max_cache_len=generation_config.max_length,
                )

        self._validate_generated_length(
            generation_config, input_ids_length, has_default_max_length
        )

        if self.device.type != input_ids.device.type:
            warnings.warn(
                "You are calling .generate() with the `input_ids` being on a device type different"
                f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
                f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
                " Please make sure that you have put `input_ids` to the"
                f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
                " running `.generate()`.",
                UserWarning,
            )

        # 8. prepare distribution pre_processing samplers
        prepared_logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_length,
            encoder_input_ids=inputs_tensor,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
            model_kwargs=model_kwargs,
        )

        # 12. expand input_ids with `num_return_sequences` additional sequences per batch
        input_ids, model_kwargs = self._expand_inputs_for_generation(
            input_ids=input_ids,
            expand_size=generation_config.num_return_sequences,
            is_encoder_decoder=self.config.is_encoder_decoder,
            **model_kwargs,
        )

        # 13. run sample
        result = self._generate(
            input_ids,
            logits_processor=prepared_logits_processor,
            max_length=generation_config.max_length,
            eos_token_id=generation_config.eos_token_id,
            do_sample=generation_config.do_sample,
            **model_kwargs,
        )
        if generation_config.cache_implementation in NEED_SETUP_CACHE_CLASSES_MAPPING:
            if not callable(getattr(self, "_reset_cache", None)):
                raise ValueError(
                    "A `static_cache` was used to generate but there was a failure when trying to  release the cache. "
                    " Make sure this model implements a `_reset_cache` function."
                )
            self._reset_cache()

        return result

    def __call__(self, *args, **kwargs) -> HeadedModelOutput:
        return self._wrapped_call_impl(*args, **kwargs)

    def _generate(
        self,
        input_ids: torch.LongTensor,
        do_sample: bool,
        logits_processor: Optional[LogitsProcessorList] = None,
        logits_warper: Optional[LogitsProcessorList] = None,
        max_length: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        ignore_heads: Optional[List[str]] = None,
        **model_kwargs,
    ) -> HeadedModelGenerateOutput:
        r"""
        Generates sequences of token ids for models with a language modeling head using **greedy decoding**.

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            max_length (`int`, *optional*, defaults to 20):
                The maximum length of the sequence to be generated.
            eos_token_id (`Union[int, List[int]]`, *optional*):
                The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
            model_kwargs:
                Additional model specific keyword arguments will be forwarded to the `forward` function of the model.
                If model is an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            HeadedModelGenerateOutput: Output of the generation including `logprobs` and `head_outputs`
        """
        assert (
            self.lm_head is not None
        ), "lm_head is not defined. Can not generate without lm_head"

        # init values
        stopping_criteria = StoppingCriteriaList()
        logits_processor = (
            logits_processor if logits_processor is not None else LogitsProcessorList()
        )
        logits_warper = (
            logits_warper if logits_warper is not None else LogitsProcessorList()
        )
        if max_length is not None:
            stopping_criteria = validate_stopping_criteria(
                stopping_criteria, max_length
            )
        if eos_token_id is not None:
            stopping_criteria.append(EosTokenCriteria(eos_token_id=eos_token_id))
        else:
            if self.generation_config.eos_token_id is not None:
                eos_token_id = self.generation_config.eos_token_id
                stopping_criteria.append(EosTokenCriteria(eos_token_id=eos_token_id))

        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]

        # init attention / hidden states / scores tuples
        logprobs = []
        head_outputs = {key: [] for key in self.heads.keys()}

        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape
        if "inputs_embeds" in model_kwargs:
            cur_len = model_kwargs["inputs_embeds"].shape[1]
        this_peer_finished = False
        unfinished_sequences = torch.ones(
            batch_size, dtype=torch.long, device=input_ids.device
        )
        model_kwargs["cache_position"] = torch.arange(cur_len, device=input_ids.device)
        eos_tensor = torch.tensor(eos_token_id, device=input_ids.device)

        while not this_peer_finished:
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # forward pass to get next token
            outputs: HeadedModelOutput = self(
                **model_inputs,
            )

            next_token_logits = outputs.preds_by_head["lm_head"][:, -1, :]
            new_logprobs = log_softmax(next_token_logits, dim=-1)

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)

            if do_sample:
                next_token_scores = logits_warper(input_ids, next_token_scores)
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            # update logprobs and head outputs
            logprobs.append(new_logprobs.gather(1, next_tokens[:, None])[:, 0])

            for key, value in outputs.preds_by_head.items():
                if key != "lm_head" and (
                    ignore_heads is None or key not in ignore_heads
                ):
                    head_outputs[key].append(value[:, -1])

            # finished sentences should have their next token be a eos token
            if eos_token_id is not None:
                next_tokens = next_tokens * unfinished_sequences + eos_tensor * (
                    1 - unfinished_sequences
                )

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(
                input_ids, None
            )
            this_peer_finished = unfinished_sequences.max() == 0

        head_outputs = {
            key: torch.stack(value, dim=1) for key, value in head_outputs.items()
        }

        return HeadedModelGenerateOutput(
            sequences=input_ids,
            logprobs=torch.stack(logprobs, dim=1),
            head_outputs=head_outputs,
            past_key_values=model_kwargs.get("past_key_values"),
        )


def get_multi_head_transformer(base_model_class: Type[PreTrainedModel]):
    """
    Patch a pretrained transformer model to add multiple heads.

    Args:
        base_model_class (Type[PreTrainedModel]): The base pretrained model class.

    Returns:
        Type[PreTrainedModel]: The new, patched class.
    """

    class TransformerWithHeads(
        HeadedModel, get_headed_pretrained_model_class(base_model_class)
    ):
        """
        Transformer model with multiple heads.

        Attributes:
            vocab_size (int): The size of the vocabulary.
            head_configs (dict[str:HeadConfig]): The configurations for the heads.
            heads (nn.ModuleDict): The heads of the model.
            lm_head (Optional[MLPHead]): The language model head.
            lm_head_config (Optional[HeadConfig]): The configuration for the language model head.
        """

        def __init__(self, config: PretrainedConfig):
            """
            Initializes the TransformerWithHeads class.

            Args:
                config (PretrainedConfig): The configurations for the headed model.
            """
            super().__init__(config)
            setattr(
                self,
                model_type_map[config.model_type][0],
                model_type_map[config.model_type][1](config.to_base_class()),
            )
            self.vocab_size: int = config.vocab_size
            self.head_configs: dict[str, HeadConfig] = {
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
            self.adaptive_loss = False
            self.adaptive_warmup = None
            self.adaptive_collect = None

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
            """
            Saves the model with all its heads to the specified directory.

            Args:
                save_directory (str | PathLike): The directory to save the model to.
                is_main_process (bool, optional): Whether the current process is the main process. Defaults to True.
                state_dict (Dict, optional): The state dictionary of the model. Defaults to None.
                save_function (Callable[..., Any], optional): The function to use to save the model. Defaults to torch.save.
                push_to_hub (bool, optional): Whether to push the model to the hub. Defaults to False.
                max_shard_size (int | str, optional): The maximum shard size. Defaults to "5GB".
                safe_serialization (bool, optional): Whether to use safe serialization. Defaults to True.
                variant (str | None, optional): The variant of the model. Defaults to None.
                token (str | bool | None, optional): The token to use for authentication. Defaults to None.
                save_peft_format (bool, optional): Whether to save in PEFT format. Defaults to True.
                **kwargs: Additional keyword arguments.
            """
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

    return TransformerWithHeads
