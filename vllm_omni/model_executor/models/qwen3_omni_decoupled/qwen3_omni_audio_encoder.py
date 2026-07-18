# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audio encoder stage for decoupled Qwen3-Omni-MoE.

Standalone stage that processes audio inputs and outputs audio embeddings.
Designed to be independently deployed and scaled.
"""

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeConfig,
)
from vllm.config import VllmConfig
from vllm.model_executor.models.qwen2_5_omni_thinker import (
    Qwen2_5OmniAudioFeatureInputs,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    maybe_prefix,
)
from vllm.logger import init_logger

from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker import (
    Qwen3OmniMoeAudioEncoder,
)
from vllm_omni.utils.nvtx import nvtx_range

logger = init_logger(__name__)


class Qwen3OmniMoeAudioEncoderStage(nn.Module):
    """Audio encoder stage: audio → audio embeddings.

    A standalone stage that loads only the audio_tower component from the
    Qwen3-Omni checkpoint. Designed for independent deployment and scaling.

    Input: audio features (input_audio_features, audio_feature_lengths)
    Output: ``embed.encoder`` items labelled ``audio`` in the standard
    inter-stage payload schema.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        config: Qwen3OmniMoeConfig = vllm_config.model_config.hf_config
        thinker_config = config.thinker_config
        quant_config = vllm_config.quant_config

        self.audio_tower = Qwen3OmniMoeAudioEncoder(
            thinker_config.audio_config,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "audio_tower"),
        )
        self.have_multimodal_outputs = True

    def _parse_audio_input(self, **kwargs) -> Qwen2_5OmniAudioFeatureInputs | None:
        """Parse and validate audio inputs from kwargs (self-contained).

        Mirrors ``Qwen2_5OmniConditionalGenerationMixin._parse_and_validate_audio_input``
        without inheriting the mixin.
        """
        input_audio_features = kwargs.pop("input_audio_features", None)
        audio_feature_lengths = kwargs.pop("audio_feature_lengths", None)
        feature_attention_mask = kwargs.pop("feature_attention_mask", None)
        if input_audio_features is None:
            return None
        if (
            input_audio_features is not None
            and isinstance(input_audio_features, torch.Tensor)
            and input_audio_features.ndim == 3
        ):
            input_audio_features = input_audio_features.reshape(-1, input_audio_features.shape[-1])
        elif input_audio_features is not None and isinstance(input_audio_features, list):
            input_audio_features = torch.cat(input_audio_features, dim=-1)
        if (
            audio_feature_lengths is not None
            and isinstance(audio_feature_lengths, torch.Tensor)
            and audio_feature_lengths.ndim == 2
        ):
            audio_feature_lengths = audio_feature_lengths.reshape(-1)
        elif audio_feature_lengths is not None and isinstance(audio_feature_lengths, list):
            audio_feature_lengths = torch.cat(audio_feature_lengths, dim=-1)
        if (
            feature_attention_mask is not None
            and isinstance(feature_attention_mask, torch.Tensor)
            and feature_attention_mask.ndim == 3
        ):
            feature_attention_mask = feature_attention_mask.reshape(-1, feature_attention_mask.shape[-1])
        elif feature_attention_mask is not None and isinstance(feature_attention_mask, list):
            for i in range(len(feature_attention_mask)):
                feature_attention_mask[i] = feature_attention_mask[i].reshape(-1)
        return Qwen2_5OmniAudioFeatureInputs(
            type="audio_features",
            input_features=input_audio_features,
            audio_feature_lengths=audio_feature_lengths,
            feature_attention_mask=feature_attention_mask,
        )

    def _process_audio(self, audio_input) -> list[torch.Tensor]:
        """Process audio through the audio tower."""
        with nvtx_range("omni_decoupled_process_audio"):
            input_features = audio_input["input_features"]
            audio_feature_lengths = audio_input["audio_feature_lengths"]

            audio_output_lengths = self._get_feat_extract_output_lengths(audio_feature_lengths)

            audio_outputs = self.audio_tower(
                input_features.to(self.audio_tower.dtype),
                feature_lens=audio_feature_lengths,
                aftercnn_lens=audio_output_lengths,
            )
            audio_features = (
                audio_outputs if isinstance(audio_outputs, torch.Tensor)
                else audio_outputs.last_hidden_state
            )

            chunks = audio_features.split(audio_output_lengths.tolist())
            return list(chunks)

    @staticmethod
    def _get_feat_extract_output_lengths(input_lengths: torch.Tensor) -> torch.Tensor:
        """Compute output lengths after convolutional feature extraction.

        Mirrors the implementation in vLLM's qwen3_omni_moe_thinker.py.
        Accounts for the audio tower's chunking with n_window=50:
        - Full windows (n_window*2=100 tokens) each produce 13 CNN output tokens.
        - The remainder tail produces fewer tokens.
        """
        input_lengths_leave = input_lengths % 100
        feat_lengths = (input_lengths_leave - 1) // 2 + 1
        output_lengths = (
            ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1
            + (input_lengths // 100) * 13
        )
        return output_lengths

    def embed_multimodal(self, **kwargs) -> list[torch.Tensor]:
        """Process multimodal inputs - only handles audio."""
        audio_input = self._parse_audio_input(**kwargs)
        if audio_input is not None:
            return self._process_audio(audio_input)

        # A visual-only request must produce no audio items.  A synthetic
        # embedding would be indistinguishable from a real audio placeholder
        # after the stage boundary and corrupt the Thinker input.
        return []

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> OmniOutput:
        """Forward pass: process audio and return embeddings."""
        with nvtx_range("omni_decoupled_forward_audio_encoder"):
            audio_embeddings = self.embed_multimodal(**kwargs)
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={
                    "embed": {"encoder": audio_embeddings},
                    "meta": {"encoder_modalities": ["audio"] * len(audio_embeddings)},
                },
            )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load only audio_tower weights from the checkpoint.

        Filters checkpoint weights for the ``thinker.audio_tower.`` prefix,
        strips it, and loads into ``self.audio_tower``.
        """
        mapper = WeightsMapper(
            orig_to_new_prefix={"thinker.audio_tower.": "audio_tower."}
        )
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["thinker.visual.", "thinker.model.", "thinker.lm_head.",
                           "talker.", "code2wav."],
        )
        loaded = loader.load_weights(weights, mapper=mapper)
        return loaded
