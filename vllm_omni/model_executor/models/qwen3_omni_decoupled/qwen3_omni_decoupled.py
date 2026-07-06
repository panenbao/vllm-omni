# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unified orchestrator for decoupled Qwen3-Omni-MoE (5-stage pipeline).

Orchestrates five independently-deployable stages:
  Stage 0: Audio Encoder  — audio → audio embeddings
  Stage 1: Visual Encoder — image/video → visual embeddings
  Stage 2: LLM            — text + embeddings → hidden states (AR generation)
  Stage 3: Talker         — hidden states → RVQ codec codes
  Stage 4: Code2Wav       — RVQ codes → audio waveform

Each stage loads only its own weight prefix from the unified checkpoint.
"""

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeCode2WavConfig,
    Qwen3OmniMoeConfig,
    Qwen3OmniMoeTalkerConfig,
)
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import (
    SupportsLoRA,
    SupportsMRoPE,
    SupportsMultiModal,
    SupportsPP,
)
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors

from vllm_omni.metrics import definitions as defs
from vllm_omni.model_executor.custom_process_mixin import CustomProcessMixin
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.model_executor.models.utils import (
    add_prefix_to_loaded_weights,
)
from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker import (
    Qwen3OmniMoeThinkerDummyInputsBuilder,
    Qwen3OmniMoeThinkerMultiModalProcessor,
    Qwen3OmniMoeThinkerProcessingInfo,
)

logger = init_logger(__name__)


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3OmniMoeThinkerMultiModalProcessor,
    info=Qwen3OmniMoeThinkerProcessingInfo,
    dummy_inputs=Qwen3OmniMoeThinkerDummyInputsBuilder,
)
class Qwen3OmniMoeDecoupledForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsPP,
    SupportsMRoPE,
    CustomProcessMixin,
):
    """
    Unified orchestrator for the decoupled Qwen3 Omni MoE (5-stage pipeline).

    Dispatches to the correct stage based on ``model_stage``.
    Each stage is independently deployable on separate GPU sets.

    Usage:
        Set ``model_stage`` to one of:
        "audio_encoder", "visual_encoder", "thinker_lm", "talker", "code2wav"
    """

    have_multimodal_outputs = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        config: Qwen3OmniMoeConfig = vllm_config.model_config.hf_config
        self.config = config
        self.model_stage = vllm_config.model_config.model_stage

        # ── Lazy-import and initialize the requested stage ────────
        if self.model_stage == "audio_encoder":
            from vllm_omni.model_executor.models.qwen3_omni_decoupled.qwen3_omni_audio_encoder import (
                Qwen3OmniMoeAudioEncoderStage,
            )
            self.stage = Qwen3OmniMoeAudioEncoderStage(
                vllm_config=vllm_config, prefix=prefix,
            )

        elif self.model_stage == "visual_encoder":
            from vllm_omni.model_executor.models.qwen3_omni_decoupled.qwen3_omni_visual_encoder import (
                Qwen3OmniMoeVisualEncoderStage,
            )
            self.stage = Qwen3OmniMoeVisualEncoderStage(
                vllm_config=vllm_config, prefix=prefix,
            )

        elif self.model_stage == "thinker_lm":
            from vllm_omni.model_executor.models.qwen3_omni_decoupled.qwen3_omni_llm import (
                Qwen3OmniMoeLLMStage,
            )
            self.stage = Qwen3OmniMoeLLMStage(
                vllm_config=vllm_config, prefix=prefix,
            )

        elif self.model_stage == "talker":
            from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_talker import (
                Qwen3OmniMoeTalkerForConditionalGeneration,
            )
            talker_vllm_config = vllm_config.with_hf_config(
                config.talker_config,
                architectures=["Qwen3OmniMoeTalkerForConditionalGeneration"],
            )
            self.stage = Qwen3OmniMoeTalkerForConditionalGeneration(
                vllm_config=talker_vllm_config, prefix=prefix,
            )
            # Wire up talker preprocess/MTP by binding the original orchestrator's
            # methods to this instance. Sets self.talker = self.stage as alias.
            import types as _types
            from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni import (
                Qwen3OmniMoeForConditionalGeneration as _Qwen3Omni,
            )
            self.talker = self.stage
            self.talker_config = config.talker_config
            # Bind methods FIRST, then set up preprocess hook
            for _name in (
                "talker_preprocess", "talker_mtp",
                "talker_preprocess_prefill", "talker_preprocess_decode",
                "_talker_cache_thinker_decode_embeds",
                "_thinker_to_talker_prefill",
                "_get_text_spk_token_id", "_get_tts_embed",
            ):
                setattr(self, _name, _types.MethodType(getattr(_Qwen3Omni, _name), self))
            self.has_preprocess = True
            self.has_postprocess = True
            self.set_custom_preprocess(self.talker_preprocess)
            # Speaker IDs for TTS voice selection
            talker_hf_config = config.talker_config
            raw_spk = getattr(talker_hf_config, "speaker_id", None) or {}
            self.tts_text_spk_token_ids = {k.lower(): v for k, v in raw_spk.items()}
            spk_default = getattr(talker_hf_config, "default_speaker", None)
            if spk_default:
                self.default_tts_text_spk_type = str(spk_default).lower()
            elif self.tts_text_spk_token_ids:
                self.default_tts_text_spk_type = list(self.tts_text_spk_token_ids.keys())[0]
            else:
                self.default_tts_text_spk_type = "default"

        elif self.model_stage == "code2wav":
            from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_code2wav import (
                Qwen3OmniMoeCode2Wav,
            )
            code2wav_vllm_config = vllm_config.with_hf_config(
                config.code2wav_config,
                architectures=["Qwen3OmniMoeCode2Wav"],
            )
            self.stage = Qwen3OmniMoeCode2Wav(
                vllm_config=code2wav_vllm_config, prefix=prefix,
            )

        else:
            raise ValueError(
                f"Invalid model_stage: {self.model_stage}. "
                f"Must be one of: 'audio_encoder', 'visual_encoder', "
                f"'thinker_lm', 'talker', 'code2wav'"
            )

        self.model = self.stage
        self.make_empty_intermediate_tensors = (
            getattr(self.stage, "make_empty_intermediate_tensors", lambda: None)
        )

    @staticmethod
    def _module_device(module: nn.Module) -> torch.device:
        return next(module.parameters(), torch.tensor(0)).device

    # ── Multimodal interface ─────────────────────────────────────

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
        *,
        is_multimodal=None,
    ) -> torch.Tensor:
        if self.model_stage in ("audio_encoder", "visual_encoder", "code2wav"):
            return torch.zeros_like(input_ids).reshape(-1, 1).repeat(1, self.vllm_config.model_config.get_hidden_size())
        if self.model_stage == "talker":
            return self.stage.embed_input_ids(input_ids)
        return self.stage.embed_input_ids(input_ids, multimodal_embeddings, is_multimodal=is_multimodal)

    def embed_multimodal(self, **kwargs):
        if hasattr(self.stage, "embed_multimodal"):
            return self.stage.embed_multimodal(**kwargs)
        return []

    def get_mm_mapping(self) -> MultiModelKeys:
        if self.model_stage in ("thinker_lm",):
            return MultiModelKeys.from_string_field(
                language_model="stage.language_model",
                connector="",
                tower_model=[],
            )
        if self.model_stage == "audio_encoder":
            return MultiModelKeys.from_string_field(
                language_model="", connector="", tower_model=["stage.audio_tower"],
            )
        if self.model_stage == "visual_encoder":
            return MultiModelKeys.from_string_field(
                language_model="", connector="", tower_model=["stage.visual"],
            )
        return MultiModelKeys.from_string_field(
            language_model="stage.language_model", connector="", tower_model=[],
        )

    # ── Forward & output wrapping ────────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> Any:
        # Code2Wav: reshape flat codes [B*16*T] → [B, 16, T] before decoding.
        # Follows the exact pattern from qwen3_omni.py:415-461 (Stage 3: Code2Wav).
        # NOTE: uses input_ids.reshape() (view, zero-copy) to avoid CPU→CUDA
        # copies during CUDA graph capture. During graph capture the runner
        # provides CUDA input_ids directly, so codes stays on CUDA.
        if self.model_stage == "code2wav":
            if input_ids is None:
                return OmniOutput(text_hidden_states=None, multimodal_outputs=None)
            seq_token_counts: list[int] | None = kwargs.get("seq_token_counts")
            if input_ids.shape[0] % 16 == 0:
                if seq_token_counts is not None:
                    max_seq_len = max(seq_token_counts) // 16
                    batch_size = len(seq_token_counts)
                    split_codes = torch.split(input_ids, seq_token_counts, dim=0)
                    codes = torch.zeros((batch_size, 16, max_seq_len),
                                        device=input_ids.device, dtype=input_ids.dtype)
                    for idx, code in enumerate(split_codes):
                        seq_len = code.shape[0] // 16
                        codes[idx, :, :seq_len] = code.reshape(16, seq_len)
                else:
                    codes = input_ids.reshape(1, 16, -1)
            else:
                if seq_token_counts is None:
                    logger.debug(
                        "Code2Wav warmup input length %s is not divisible by 16; padding with zeros.",
                        input_ids.shape[0],
                    )
                input_ids_flatten = input_ids.reshape(-1)
                pad_len = 16 - input_ids.shape[0] % 16
                if pad_len < 16:
                    input_ids_flatten = torch.cat([
                        input_ids_flatten,
                        torch.zeros(pad_len, dtype=torch.long, device=input_ids.device),
                    ])
                codes = input_ids_flatten.reshape(1, 16, -1)
            codes = codes.to(dtype=torch.long)
            left_context_size = self._extract_code2wav_left_context(kwargs)
            if self.vllm_config.model_config.async_chunk:
                return self.stage.chunked_decode_streaming(
                    codes, left_context_size=left_context_size,
                    seq_token_counts=seq_token_counts,
                )
            return self.stage.chunked_decode(
                codes, chunk_size=300, left_context_size=25,
                seq_token_counts=seq_token_counts,
            )

        return self.stage.forward(
            input_ids=input_ids, positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds, **kwargs,
        )

    @staticmethod
    def _extract_code2wav_left_context(kwargs: dict) -> list[int]:
        """Extract left_context_size from runtime additional info."""
        runtime_info = kwargs.get("runtime_additional_information")
        if runtime_info is None:
            return [0]
        ctx = []
        for info in runtime_info:
            meta = info.get("meta", {})
            if "left_context_size" in meta:
                ctx.append(meta["left_context_size"])
        return ctx or [0]

    def make_omni_output(self, model_outputs: Any, **kwargs) -> OmniOutput:
        """Wrap stage outputs in ``OmniOutput``.

        Called by ``gpu_model_runner`` after ``forward()``.
        """
        if isinstance(model_outputs, OmniOutput):
            return model_outputs

        if self.model_stage in ("audio_encoder", "visual_encoder"):
            return OmniOutput(text_hidden_states=None, multimodal_outputs=None)

        if self.model_stage == "thinker_lm":
            text_hidden_states, captured_dict = model_outputs
            multimodal: dict[str, Any] = captured_dict or {}
            try:
                tts_tokens = torch.tensor(
                    [[151672, 151673, 151671]],
                    device=text_hidden_states.device, dtype=torch.long,
                )
                tts_embeds = self.stage.embed_input_ids(tts_tokens)
                if isinstance(tts_embeds, torch.Tensor) and tts_embeds.ndim == 3 and tts_embeds.shape[1] == 3:
                    bos, eos, pad = tts_embeds.to(text_hidden_states.device).chunk(3, dim=1)
                    embed = multimodal.setdefault("embed", {})
                    embed["tts_bos"] = [bos]
                    embed["tts_eos"] = [eos]
                    embed["tts_pad"] = [pad]
            except Exception:
                pass
            return OmniOutput(
                text_hidden_states=text_hidden_states.reshape(-1, text_hidden_states.shape[-1]),
                multimodal_outputs=multimodal,
            )

        if self.model_stage == "talker":
            talker_hidden = model_outputs
            info_dicts = kwargs.get("model_intermediate_buffer")
            if info_dicts is None:
                info_dicts = kwargs.get("runtime_additional_information")
            code_predictor_codes = [
                info.get("codes", {}).get("audio") for info in (info_dicts or [])
            ]
            if code_predictor_codes:
                audio_codes = torch.cat(code_predictor_codes, dim=0)
                span_len = audio_codes.shape[0]
                return OmniOutput(
                    text_hidden_states=talker_hidden[:span_len],
                    multimodal_outputs={"codes": {"audio": audio_codes}},
                )
            return OmniOutput(
                text_hidden_states=talker_hidden,
                multimodal_outputs=None,
            )

        if self.model_stage == "code2wav":
            audio_tensors = model_outputs
            sample_rate = defs.resolve_audio_sample_rate(self.stage.config)
            sr = [torch.tensor(sample_rate, dtype=torch.int32) for _ in audio_tensors]
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={
                    "model_outputs": [t.reshape(1, -1) for t in audio_tensors],
                    "sr": sr,
                },
            )

        return model_outputs

    # ── Sampling ─────────────────────────────────────────────────

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: Any = None,
    ) -> torch.Tensor | None:
        # if hasattr(self.stage, "compute_logits"):
        #     return self.stage.compute_logits(hidden_states)
        if (
            getattr(self, "model_stage", None) == "talker"
            and sampling_metadata is not None
            and (sampling_metadata.temperature is None)
        ):
            self._warn_talker_sampling_temperature(sampling_metadata)
        return self.stage.compute_logits(hidden_states)

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: Any,
    ) -> Any:
        if hasattr(self.stage, "sample"):
            return self.stage.sample(logits, sampling_metadata)
        return self.model.sample(logits, sampling_metadata)

    # ── M-RoPE ───────────────────────────────────────────────────

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[Any] | None = None,
        **kwargs: object,
    ) -> tuple[torch.Tensor, int]:
        if hasattr(self.stage, "get_mrope_input_positions"):
            return self.stage.get_mrope_input_positions(input_tokens, mm_features)
        seq_len = len(input_tokens)
        return torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(3, seq_len), 0

    def get_language_model(self) -> nn.Module | None:
        if self.model_stage == "thinker_lm":
            return self.stage.language_model
        return getattr(self.model, "language_model", None)

    # ── Weight loading ───────────────────────────────────────────

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights for the active stage.

        Each stage loads only its own weight prefix from the checkpoint.
        Adds ``stage.`` prefix to returned names to match ``self.stage``
        in the orchestrator's module hierarchy.
        """
        loaded = self.stage.load_weights(weights)
        return add_prefix_to_loaded_weights(loaded, "stage")
