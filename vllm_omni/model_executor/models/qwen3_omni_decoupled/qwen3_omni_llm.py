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
from vllm.model_executor.models.qwen2_5_omni_thinker import (
    check_interleaved_audio_video,
    merge_interleaved_embeddings,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    _merge_multimodal_embeddings,
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
from vllm_omni.utils.nvtx import nvtx_range
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

        # Deepstack buffer
        _vision_cfg = getattr(thinker_config, "vision_config", None)
        self._ds_idx = getattr(_vision_cfg, "deepstack_visual_indexes", None) if _vision_cfg else None
        if self._ds_idx is not None:
            self._ds_num_level = len(self._ds_idx)
            self._ds_input_embeds: list[torch.Tensor] = [
                torch.zeros(
                    vllm_config.scheduler_config.max_num_batched_tokens,
                    thinker_config.text_config.hidden_size,
                )
                for _ in range(self._ds_num_level)
            ]
            self._ds_input_embeds_num_tokens = 0
        else:
            self._ds_num_level = 0
            self._ds_input_embeds = []
            self._ds_input_embeds_num_tokens = 0

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

    # ── Deepstack helpers (mirrors fused Qwen3OmniMoeThinker) ──────

    def _ds_set_input_embeds(self, deepstack_input_embeds: torch.Tensor) -> None:
        """Store deepstack input embeddings."""
        if not self._ds_input_embeds:
            return
        num_tokens = deepstack_input_embeds.size(1)
        if num_tokens > self._ds_input_embeds[0].size(0):
            self._ds_resize(num_tokens)
        for idx in range(self._ds_num_level):
            self._ds_input_embeds[idx][:num_tokens].copy_(deepstack_input_embeds[idx])
        self._ds_input_embeds_num_tokens = num_tokens

    def _ds_resize(self, num_tokens: int) -> None:
        for idx in range(self._ds_num_level):
            new_buf = torch.zeros(num_tokens, self._ds_input_embeds[0].shape[-1],
                                  dtype=self._ds_input_embeds[0].dtype,
                                  device=self._ds_input_embeds[0].device)
            new_buf[:self._ds_input_embeds[idx].size(0)] = self._ds_input_embeds[idx]
            self._ds_input_embeds[idx] = new_buf

    def _ds_get_input_embeds(self, num_tokens: int) -> IntermediateTensors | None:
        """Get deepstack input embeddings as IntermediateTensors."""
        if not self._ds_input_embeds:
            return None
        if num_tokens > self._ds_input_embeds[0].size(0):
            self._ds_resize(num_tokens)
        n_valid = self._ds_input_embeds_num_tokens
        if num_tokens > n_valid:
            for idx in range(self._ds_num_level):
                self._ds_input_embeds[idx][n_valid:num_tokens].zero_()
        return IntermediateTensors({
            f"deepstack_input_embeds_{idx}": self._ds_input_embeds[idx][:num_tokens]
            for idx in range(self._ds_num_level)
        })

    def _ds_clear(self, num_tokens: int) -> None:
        """Zero out consumed deepstack entries."""
        if not self._ds_input_embeds:
            return
        if self._ds_input_embeds_num_tokens == 0:
            return
        if num_tokens > 0:
            clear_len = min(num_tokens, self._ds_input_embeds[0].size(0))
            for idx in range(self._ds_num_level):
                self._ds_input_embeds[idx][:clear_len].zero_()
            self._ds_input_embeds_num_tokens = 0

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Embed input IDs, merging with cached encoder embeddings.

        Uses the same ``_merge_multimodal_embeddings`` logic as the native
        Qwen3OmniMoeThinker for precise per-modality placement.
        """
        inputs_embeds = self.language_model.embed_input_ids(input_ids)

        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds

        # The native implementation splits visual embeddings in place.
        # Connector-backed stages may provide a tuple, so normalize it first.
        multimodal_embeddings = list(multimodal_embeddings)

        # Split and scatter deepstack features exactly as the fused thinker does.
        # In particular, the deepstack tensor must be aligned to the complete
        # scheduled token sequence; packing visual features at the beginning of
        # the buffer injects them into text/audio tokens and corrupts the KV cache.
        _text_hidden = self.config.text_config.hidden_size
        _vision_cfg = getattr(self.config, "vision_config", None)
        _ds_idx = getattr(_vision_cfg, "deepstack_visual_indexes", None) if _vision_cfg else None
        is_mm_device = (
            is_multimodal.to(device=input_ids.device, non_blocking=True)
            if is_multimodal is not None
            else None
        )
        is_video = (
            is_mm_device & (input_ids == self.config.video_token_id)
            if is_mm_device is not None
            else None
        )
        is_audio = (
            is_mm_device & (input_ids == self.config.audio_token_id)
            if is_mm_device is not None
            else None
        )
        num_video = int(is_video.sum().item()) if is_video is not None else 0
        num_audio = int(is_audio.sum().item()) if is_audio is not None else 0
        is_interleaved = (
            check_interleaved_audio_video(is_video, is_audio, num_video, num_audio)
            if is_video is not None and is_audio is not None
            else False
        )

        has_vision_embeddings = [
            emb.ndim == 2 and emb.shape[-1] != _text_hidden
            for emb in multimodal_embeddings
        ]
        if _ds_idx is not None and any(has_vision_embeddings) and is_mm_device is not None:
            _ds_len = len(_ds_idx)
            _ds_multiscale: list[torch.Tensor] = []

            if is_interleaved:
                is_vision = is_video.clone()
            else:
                is_vision = torch.zeros_like(is_mm_device)
                mm_positions = torch.nonzero(is_mm_device, as_tuple=True)[0]
                mm_position_idx = 0

            for i, emb in enumerate(multimodal_embeddings):
                num_tokens = emb.shape[0]
                if emb.ndim == 2 and emb.shape[-1] != _text_hidden:
                    _vis_dim = emb.shape[-1] // (_ds_len + 1)
                    _multi_dim = _vis_dim * _ds_len
                    _emb_main, _emb_multi = torch.split(emb, [_vis_dim, _multi_dim], dim=-1)
                    multimodal_embeddings[i] = _emb_main
                    _ds_multiscale.append(_emb_multi)
                    if not is_interleaved:
                        current_positions = mm_positions[
                            mm_position_idx : mm_position_idx + num_tokens
                        ]
                        is_vision[current_positions] = True
                elif not is_interleaved:
                    current_positions = mm_positions[
                        mm_position_idx : mm_position_idx + num_tokens
                    ]
                    is_vision[current_positions] = False

                if not is_interleaved:
                    mm_position_idx += num_tokens

            if _ds_multiscale:
                deepstack_input_embeds = inputs_embeds.new_zeros(
                    inputs_embeds.size(0), _ds_len * inputs_embeds.size(1)
                )
                deepstack_input_embeds = _merge_multimodal_embeddings(
                    inputs_embeds=deepstack_input_embeds,
                    multimodal_embeddings=_ds_multiscale,
                    is_multimodal=is_vision,
                )
                deepstack_input_embeds = (
                    deepstack_input_embeds.view(
                        inputs_embeds.shape[0], _ds_len, _vis_dim
                    )
                    .permute(1, 0, 2)
                    .contiguous()
                )
                self._ds_set_input_embeds(deepstack_input_embeds)

        if is_interleaved:
            return merge_interleaved_embeddings(
                inputs_embeds,
                multimodal_embeddings,
                is_video,
                is_audio,
                is_mm_device,
                num_video,
                num_audio,
            )

        if is_multimodal is not None:
            try:
                _result = _merge_multimodal_embeddings(
                    inputs_embeds=inputs_embeds,
                    multimodal_embeddings=multimodal_embeddings,
                    is_multimodal=is_multimodal,
                )
            except (ValueError, RuntimeError):
                logger.warning(
                    "[LLM_MERGE] count mismatch in _merge_multimodal_embeddings, "
                    "falling back to token-ID scatter (warmup?)",
                    exc_info=True,
                )
                _result = None
            if _result is not None:
                return _result

        # Fallback: token-ID-based scatter for warmup/profile.
        is_mm_device = is_multimodal.to(device=input_ids.device, non_blocking=True)
        mm_mask = is_mm_device & (
            (input_ids == self.config.audio_token_id)
            | (input_ids == self.config.video_token_id)
            | (input_ids == self.config.image_token_id)
        )
        if mm_mask.any():
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
        with nvtx_range(f"omni_decoupled_forward_thinker_llm"):
            capture_kwargs: dict[str, Any] = {}
            if self._accept_hidden_layer is not None:
                capture_kwargs = {
                    "capture_layer_indices": [0, int(self._accept_hidden_layer)],
                    "return_hidden_states": True,
                }
            ds_input = self._ds_get_input_embeds(inputs_embeds.size(0)) if inputs_embeds is not None else None
            if ds_input is not None:
                capture_kwargs["deepstack_input_embeds"] = ds_input

            hidden_states, captured_hidden_states = self.language_model.model(
                input_ids,
                positions,
                intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **capture_kwargs,
            )

            if inputs_embeds is not None:
                self._ds_clear(inputs_embeds.size(0))

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
        """Compute M-RoPE position IDs for multimodal inputs."""
        seq_len = len(input_tokens)
        if mm_features is None:
            linear = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(3, seq_len)
            return linear, 0

        import numpy as np

        from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker import (
            _get_feat_extract_output_lengths,
        )

        vision_cfg = getattr(self.config, "vision_config", None)
        spatial_merge_size = getattr(vision_cfg, "spatial_merge_size", 2)
        position_id_per_seconds = getattr(self.config, "position_id_per_seconds", 1)

        sorted_features = sorted(mm_features, key=lambda f: f.mm_position.offset)
        llm_pos_ids_list: list[np.ndarray] = []
        st = 0

        for mm_feature in sorted_features:
            offset = mm_feature.mm_position.offset
            modality = mm_feature.modality

            text_len = offset - st
            st_idx = int(llm_pos_ids_list[-1].max()) + 1 if llm_pos_ids_list else 0

            if text_len > 0:
                llm_pos_ids_list.append(np.broadcast_to(np.arange(text_len), (3, text_len)) + st_idx)
                st_idx += text_len

            bos_pos = np.broadcast_to(np.array([st_idx]), (3, 1))
            llm_pos_ids_list.append(bos_pos)
            st_idx += 1

            if modality == "audio":
                fl_elem = mm_feature.data.get("audio_feature_length")
                if fl_elem is None:
                    fl_elem = mm_feature.data.get("audio_feature_lengths")
                fl = int(fl_elem.data.item()) if hasattr(fl_elem, 'data') else int(fl_elem)
                audio_tokens = int(_get_feat_extract_output_lengths(torch.tensor([fl])).item())
                audio_pos = np.broadcast_to(np.arange(audio_tokens), (3, audio_tokens)) + st_idx
                llm_pos_ids_list.append(audio_pos)
                st_idx = int(audio_pos.max()) + 1
                eos_pos = np.broadcast_to(np.array([st_idx]), (3, 1))
                llm_pos_ids_list.append(eos_pos)
                st = offset + 1 + audio_tokens + 1

            elif modality == "image":
                grid_thw = mm_feature.data["image_grid_thw"].data
                t, h, w = grid_thw.tolist()
                h = h // spatial_merge_size
                w = w // spatial_merge_size
                t_factor = position_id_per_seconds
                grid_indices = np.indices((t, h, w))
                if t_factor != 1.0:
                    grid_indices[0] = (grid_indices[0] * t_factor).astype(np.int64)
                llm_pos_ids_list.append(grid_indices.reshape(3, -1) + st_idx)
                image_len = t * h * w
                st_idx = int(llm_pos_ids_list[-1].max()) + 1
                eos_pos = np.broadcast_to(np.array([st_idx]), (3, 1))
                llm_pos_ids_list.append(eos_pos)
                st = offset + 1 + image_len + 1

            elif modality == "video":
                grid_thw = mm_feature.data["video_grid_thw"].data
                t, h, w = grid_thw.tolist()
                h = h // spatial_merge_size
                w = w // spatial_merge_size
                second_per_grid_ts = 2.0
                t_factor = second_per_grid_ts * position_id_per_seconds
                grid_indices = np.indices((t, h, w))
                grid_indices[0] = (grid_indices[0] * t_factor).astype(np.int64)
                llm_pos_ids_list.append(grid_indices.reshape(3, -1) + st_idx)
                video_len = t * h * w
                st_idx = int(llm_pos_ids_list[-1].max()) + 1
                eos_pos = np.broadcast_to(np.array([st_idx]), (3, 1))
                llm_pos_ids_list.append(eos_pos)
                st = offset + 1 + video_len + 1

        if st < seq_len:
            text_len = seq_len - st
            st_idx = int(llm_pos_ids_list[-1].max()) + 1 if llm_pos_ids_list else 0
            llm_pos_ids_list.append(np.broadcast_to(np.arange(text_len), (3, text_len)) + st_idx)

        llm_pos_ids = np.concatenate(llm_pos_ids_list, axis=1)
        llm_pos_ids = torch.from_numpy(llm_pos_ids)
        mrope_position_delta = int(llm_pos_ids.max()) + 1 - seq_len
        return llm_pos_ids, mrope_position_delta

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
