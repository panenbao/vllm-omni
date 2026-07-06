# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Language model stage for decoupled Qwen3-Omni-MoE.

Standalone stage that runs the AR language model, receiving pre-computed
encoder embeddings from upstream audio/visual encoder stages.
Designed to be independently deployed and scaled.
"""

from collections.abc import Iterable, Sequence
from dataclasses import replace
from typing import Any

import torch
import torch.nn as nn
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeConfig,
)
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import (
    SupportsLoRA,
    SupportsMRoPE,
    SupportsPP,
    MultiModalEmbeddings,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker import (
    Qwen3MoeLLMForCausalLM,
)
from vllm_omni.quantization.component_config import (
    PRE_QUANTIZED_METHODS,
    ComponentQuantizationConfig,
)

logger = init_logger(__name__)


class Qwen3OmniMoeLLMStage(nn.Module):
    """Language model stage: text + encoder embeddings → hidden states (AR generation).

    A standalone stage that loads only the language model and lm_head from the
    Qwen3-Omni checkpoint. Receives pre-computed multimodal embeddings from
    upstream audio/visual encoder stages via ``additional_information``.

    Input: text token IDs + optional pre-computed encoder embeddings
    Output: OmniOutput with text_hidden_states + captured layer dict for talker
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        config: Qwen3OmniMoeConfig = vllm_config.model_config.hf_config
        thinker_config = config.thinker_config
        quant_config = vllm_config.quant_config
        self.config = thinker_config

        # Quantization: only the language model is quantized
        if isinstance(quant_config, ComponentQuantizationConfig):
            language_quant = quant_config.resolve("language_model")
        elif quant_config is not None:
            if quant_config.get_name() in PRE_QUANTIZED_METHODS:
                language_quant = quant_config
            else:
                quant_config = ComponentQuantizationConfig(
                    component_configs={"language_model": quant_config},
                    default_config=None,
                )
                vllm_config = replace(vllm_config, quant_config=quant_config)
                language_quant = quant_config.resolve("language_model")
        else:
            language_quant = None

        lm_vllm_config = vllm_config.with_hf_config(
            thinker_config.text_config,
            architectures=["Qwen3MoeForCausalLM"],
        )
        if language_quant is not quant_config:
            lm_vllm_config = replace(lm_vllm_config, quant_config=language_quant)

        self.language_model = Qwen3MoeLLMForCausalLM(
            vllm_config=lm_vllm_config,
            prefix=maybe_prefix(prefix, "language_model"),
        )
        self.make_empty_intermediate_tensors = self.language_model.make_empty_intermediate_tensors
        self.sampler = Sampler()
        self.have_multimodal_outputs = True

        # Cached encoder embeddings from upstream stages
        self._cached_encoder_embeddings: list[torch.Tensor] | None = None
        self._accept_hidden_layer = getattr(config.talker_config, "accept_hidden_layer", 24)

    def set_encoder_embeddings(self, embeddings: list[torch.Tensor] | None) -> None:
        """Set pre-computed encoder embeddings from upstream stages."""
        self._cached_encoder_embeddings = embeddings

    def embed_multimodal(self, **kwargs) -> MultiModalEmbeddings | None:
        """Return cached encoder embeddings if available.

        Falls back to a dummy for profile/sanity-check runs (upstream
        encoder stages haven't run yet, so no cached embeddings exist).
        """
        if self._cached_encoder_embeddings is not None:
            return tuple(self._cached_encoder_embeddings)
        # Profile run: return dummy to pass sanity_check_mm_encoder_outputs
        if any(v is not None for v in kwargs.values()):
            return [torch.zeros(1, self.config.text_config.hidden_size,
                                dtype=torch.bfloat16)]
        return []

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Embed input IDs, merging with cached encoder embeddings."""
        inputs_embeds = self.language_model.embed_input_ids(input_ids)

        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds

        # Simple merge: replace placeholder token embeddings with encoder embeddings.
        # For full interleaved support, use the thinker's merge logic; here we
        # use a simpler position-based replacement since the upstream encoders
        # have already determined the embedding positions.
        if is_multimodal is not None:
            is_mm_device = is_multimodal.to(device=input_ids.device, non_blocking=True)
            mm_mask = is_mm_device & (
                (input_ids == self.config.audio_token_id)
                | (input_ids == self.config.video_token_id)
                | (input_ids == self.config.image_token_id)
            )
            if mm_mask.any():
                # Replace embeddings at multimodal positions
                mm_indices = mm_mask.nonzero(as_tuple=False).squeeze(-1)
                emb_flat = torch.cat([e.to(dtype=inputs_embeds.dtype, device=inputs_embeds.device)
                                      for e in multimodal_embeddings], dim=0)
                num_avail = min(len(mm_indices), emb_flat.shape[0])
                if num_avail > 0:
                    inputs_embeds[mm_indices[:num_avail]] = emb_flat[:num_avail]

        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Forward pass through the language model.

        Captures layer 0 and accept_hidden_layer for talker conditioning.
        """
        capture_kwargs = {}
        if self._accept_hidden_layer is not None:
            capture_kwargs = {
                "capture_layer_indices": [0, int(self._accept_hidden_layer)],
                "return_hidden_states": True,
            }

        hidden_states, captured_hidden_states = self.language_model.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **capture_kwargs,
        )

        return hidden_states, captured_hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        """Compute logits from hidden states."""
        return self.language_model.compute_logits(hidden_states)

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput:
        """Sample tokens from logits."""
        return self.sampler(logits, sampling_metadata)

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[Any] | None = None,
    ) -> tuple[torch.Tensor, int]:
        """Compute M-RoPE position IDs for multimodal inputs.

        Falls back to linear positions when no mm_features (text-only).
        """
        if mm_features is None:
            seq_len = len(input_tokens)
            linear = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(3, seq_len)
            return linear, 0
        # M-RoPE with multimodal features - use the thinker's computation
        from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker import (
            Qwen3OmniMoeThinkerForConditionalGeneration,
        )
        return Qwen3OmniMoeThinkerForConditionalGeneration.get_mrope_input_positions(
            None, input_tokens, mm_features
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load only language model weights from the checkpoint.

        Filters checkpoint weights for ``thinker.model.`` and ``thinker.lm_head.``
        prefixes, remaps them, and loads into ``self.language_model``.
        """
        mapper = WeightsMapper(
            orig_to_new_prefix={
                "thinker.lm_head.": "language_model.lm_head.",
                "thinker.model.": "language_model.model.",
            }
        )
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["thinker.audio_tower.", "thinker.visual.",
                           "talker.", "code2wav."],
        )
        loaded_weights = loader.load_weights(weights, mapper=mapper)
        return loaded_weights
