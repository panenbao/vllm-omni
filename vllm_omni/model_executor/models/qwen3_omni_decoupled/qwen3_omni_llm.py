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
        """Embed input IDs, merging with cached encoder embeddings.

        Uses the same ``_merge_multimodal_embeddings`` logic as the native
        Qwen3OmniMoeThinker for precise per-modality placement.
        """
        inputs_embeds = self.language_model.embed_input_ids(input_ids)

        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds

        # # [DEBUG] Stage-2 gather before deepstack split: text embed + multimodal embed stats
        # logger.info("[DBG_MERGE] Stage-2 text_embed shape=%s mean=%s std=%s",
        #             inputs_embeds.shape, inputs_embeds.float().mean().item(),
        #             inputs_embeds.float().std().item())
        # for i, emb in enumerate(multimodal_embeddings):
        #     _tag = "visual" if (emb.ndim == 2 and emb.shape[-1] != inputs_embeds.shape[-1]) else "audio"
        #     logger.info("[DBG_MERGE] Stage-2 mm_before[%d](%s) shape=%s mean=%s std=%s",
        #                 i, _tag, emb.shape, emb.float().mean().item(), emb.float().std().item())

        # [Decoupled] Split deepstack visual embeddings to match LM hidden_size.
        _text_hidden = self.config.text_config.hidden_size
        _vision_cfg = getattr(self.config, "vision_config", None)
        _ds_idx = getattr(_vision_cfg, "deepstack_visual_indexes", None) if _vision_cfg else None
        if _ds_idx is not None:
            _ds_len = len(_ds_idx)
            for i, emb in enumerate(multimodal_embeddings):
                if emb.ndim == 2 and emb.shape[-1] != _text_hidden:
                    _vis_dim = emb.shape[-1] // (_ds_len + 1)
                    multimodal_embeddings[i] = emb[:, :_vis_dim]
                    # logger.info("[DBG_MERGE] Stage-2 mm_after_split[%d] shape=%s mean=%s std=%s",
                    #             i, multimodal_embeddings[i].shape,
                    #             multimodal_embeddings[i].float().mean().item(),
                    #             multimodal_embeddings[i].float().std().item())

        # # [DEBUG] Log per-modality token positions
        # _at = getattr(self.config, "audio_token_id", None)
        # _vt = getattr(self.config, "video_token_id", None)
        # _it = getattr(self.config, "image_token_id", None)
        # for _tid, _nm in [(_vt, "video"), (_at, "audio"), (_it, "image")]:
        #     if _tid is None:
        #         continue
        #     _m = input_ids == _tid
        #     if _m.any():
        #         _pos = _m.nonzero(as_tuple=False).squeeze(-1).tolist()
        #         if isinstance(_pos, list) and _pos:
        #             logger.info("[POS_MAP] Stage-2 %s token_positions=[%d..%d] count=%d",
        #                         _nm, _pos[0], _pos[-1], len(_pos))

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
            # # [DEBUG] Log per-token embedding stats for the full merged sequence
            # else:
            #     _tok_id_map = {}
            #     _at = getattr(self.config, "audio_token_id", None)
            #     _vt = getattr(self.config, "video_token_id", None)
            #     _it = getattr(self.config, "image_token_id", None)
            #     for _tid, _nm in [(_vt, "V"), (_at, "A"), (_it, "I")]:
            #         if _tid is not None:
            #             _m = input_ids == _tid
            #             if _m.any():
            #                 _tok_id_map[_tid] = _nm
            #     for _pos in range(_result.shape[0]):
            #         _tok = input_ids[_pos].item()
            #         _tag = _tok_id_map.get(_tok, "T")
            #         _v = _result[_pos].float()
            #         logger.info("[EMB_DBG] Stage-2 pos=%5d tag=%s tok=%6d mean=%9.6f std=%9.6f min=%9.6f max=%9.6f",
            #                     _pos, _tag, _tok, _v.mean().item(), _v.std().item(),
            #                     _v.min().item(), _v.max().item())
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

        # # [DEBUG] Log per-token stats for the full sequence (fallback path)
        # _tok_id_map = {}
        # for _tid, _nm in [(_vt, "V"), (_at, "A"), (_it, "I")]:
        #     if _tid is not None:
        #         _m = input_ids == _tid
        #         if _m.any():
        #             _tok_id_map[_tid] = _nm
        # for _pos in range(inputs_embeds.shape[0]):
        #     _tok = input_ids[_pos].item()
        #     _tag = _tok_id_map.get(_tok, "T")
        #     _v = inputs_embeds[_pos].float()
        #     logger.info("[EMB_DBG] Stage-2 pos=%5d tag=%s tok=%6d mean=%9.6f std=%9.6f min=%9.6f max=%9.6f",
        #                 _pos, _tag, _tok, _v.mean().item(), _v.std().item(),
        #                 _v.min().item(), _v.max().item())

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
        return llm_pos_ids, 0

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
