# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Visual encoder stage for decoupled Qwen3-Omni-MoE.

Standalone stage that processes image/video inputs and outputs visual embeddings.
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
from vllm.logger import init_logger
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    maybe_prefix,
)
from vllm.model_executor.models.qwen2_5_vl import (
    Qwen2_5_VLImagePixelInputs,
    Qwen2_5_VLVideoPixelInputs,
)

from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker import (
    Qwen3Omni_VisionTransformer,
)
from vllm_omni.utils.nvtx import nvtx_range

logger = init_logger(__name__)


class Qwen3OmniMoeVisualEncoderStage(nn.Module):
    """Visual encoder stage: image/video → visual embeddings.

    A standalone stage that loads only the visual (vision transformer) component
    from the Qwen3-Omni checkpoint.

    Input: pixel_values (+ image_grid_thw for images, pixel_values_videos + video_grid_thw for videos)
    Output: ``embed.encoder`` items labelled ``image`` / ``video``.  It also
    carries the upstream audio items through to the Thinker stage.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        config: Qwen3OmniMoeConfig = vllm_config.model_config.hf_config
        thinker_config = config.thinker_config
        quant_config = vllm_config.quant_config

        self.visual = Qwen3Omni_VisionTransformer(
            vision_config=thinker_config.vision_config,
            norm_eps=getattr(thinker_config.text_config, "rms_norm_eps", 1e-6),
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "visual"),
        )
        self.have_multimodal_outputs = True

        # Deepstack support
        self.use_deepstack = hasattr(thinker_config.vision_config, "deepstack_visual_indexes")

    def _parse_image_input(self, **kwargs) -> Qwen2_5_VLImagePixelInputs | None:
        """Parse image inputs from kwargs."""
        pixel_values = kwargs.get("pixel_values", None)
        image_grid_thw = kwargs.get("image_grid_thw", None)
        if pixel_values is None:
            return None
        if isinstance(pixel_values, torch.Tensor) and pixel_values.ndim == 3:
            pixel_values = pixel_values.reshape(-1, pixel_values.shape[-1])
        if isinstance(image_grid_thw, torch.Tensor) and image_grid_thw.ndim == 3:
            image_grid_thw = image_grid_thw.reshape(-1, image_grid_thw.shape[-1])
        return Qwen2_5_VLImagePixelInputs(
            type="pixel_values",
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )

    def _parse_video_input(self, **kwargs) -> Qwen2_5_VLVideoPixelInputs | None:
        """Parse video inputs from kwargs."""
        pixel_values_videos = kwargs.get("pixel_values_videos", None)
        video_grid_thw = kwargs.get("video_grid_thw", None)
        if pixel_values_videos is None:
            return None
        if isinstance(pixel_values_videos, torch.Tensor) and pixel_values_videos.ndim == 3:
            pixel_values_videos = pixel_values_videos.reshape(-1, pixel_values_videos.shape[-1])
        if isinstance(video_grid_thw, torch.Tensor) and video_grid_thw.ndim == 3:
            video_grid_thw = video_grid_thw.reshape(-1, video_grid_thw.shape[-1])
        return Qwen2_5_VLVideoPixelInputs(
            type="pixel_values_videos",
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
        )

    def _process_image(self, image_input: Qwen2_5_VLImagePixelInputs) -> list[torch.Tensor]:
        """Process image through the vision transformer."""
        grid_thw = image_input["image_grid_thw"]
        assert grid_thw.ndim == 2
        pixel_values = image_input["pixel_values"].type(self.visual.dtype)
        image_embeds = self.visual(pixel_values, grid_thw=grid_thw)
        merge_size = self.visual.spatial_merge_size
        sizes = grid_thw.prod(-1) // merge_size // merge_size
        return list(image_embeds.split(sizes.tolist()))

    def _process_video(self, video_input: Qwen2_5_VLVideoPixelInputs) -> list[torch.Tensor]:
        """Process video through the vision transformer."""
        grid_thw = video_input["video_grid_thw"]
        assert grid_thw.ndim == 2
        pixel_values = video_input["pixel_values_videos"].type(self.visual.dtype)
        video_embeds = self.visual(pixel_values, grid_thw=grid_thw)
        merge_size = self.visual.spatial_merge_size
        sizes = grid_thw.prod(-1) // merge_size // merge_size
        return list(video_embeds.split(sizes.tolist()))

    def _embed_multimodal_with_modalities(self, **kwargs) -> tuple[list[torch.Tensor], list[str]]:
        """Process multimodal inputs - handles image and video."""
        embeddings: list[torch.Tensor] = []
        modalities: list[str] = []
        image_input = self._parse_image_input(**kwargs)
        if image_input is not None:
            image_embeddings = self._process_image(image_input)
            embeddings.extend(image_embeddings)
            modalities.extend(["image"] * len(image_embeddings))
        video_input = self._parse_video_input(**kwargs)
        if video_input is not None:
            video_embeddings = self._process_video(video_input)
            embeddings.extend(video_embeddings)
            modalities.extend(["video"] * len(video_embeddings))
        return embeddings, modalities

    def embed_multimodal(self, **kwargs) -> list[torch.Tensor]:
        """vLLM multimodal hook; metadata is carried by ``forward`` only."""
        embeddings, _ = self._embed_multimodal_with_modalities(**kwargs)
        return embeddings

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> OmniOutput:
        """Forward pass: process image/video and return embeddings."""
        with nvtx_range("omni_decoupled_forward_visual_encoder"):
            visual_embeddings, visual_modalities = self._embed_multimodal_with_modalities(**kwargs)

            # Stage 0 may have produced audio embeddings.  Stage 1 must carry
            # them forward together with its visual results; otherwise audio
            # conditioning is silently lost before the Thinker stage.
            upstream_embeddings: list[torch.Tensor] = []
            upstream_modalities: list[str] = []
            for info in kwargs.get("runtime_additional_information") or kwargs.get("model_intermediate_buffer") or []:
                if not isinstance(info, dict):
                    continue
                embed = info.get("embed", {})
                meta = info.get("meta", {})
                values = embed.get("encoder", []) if isinstance(embed, dict) else []
                labels = meta.get("encoder_modalities", []) if isinstance(meta, dict) else []
                if not values:
                    values = info.get("embed.encoder", [])
                if not labels:
                    labels = info.get("meta.encoder_modalities", [])
                if isinstance(values, torch.Tensor):
                    values = [values]
                if isinstance(values, list) and len(values) == len(labels):
                    upstream_embeddings.extend(v for v in values if isinstance(v, torch.Tensor))
                    upstream_modalities.extend(labels)

            encoder_embeddings = upstream_embeddings + visual_embeddings
            encoder_modalities = upstream_modalities + visual_modalities
            logger.debug("Decoupled visual stage produced %d encoder items", len(encoder_embeddings))
            # # [DEBUG] Stage-1 send: per-modality embed stats
            # for i, (emb, mod) in enumerate(zip(encoder_embeddings, encoder_modalities)):
            #     logger.info("[DBG_SEND] Stage-1 %s[%d] shape=%s mean=%s std=%s",
            #                 mod, i, emb.shape, emb.float().mean().item(), emb.float().std().item())
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={
                    "embed": {"encoder": encoder_embeddings},
                    "meta": {"encoder_modalities": encoder_modalities},
                },
            )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load only visual transformer weights from the checkpoint.

        Filters checkpoint weights for the ``thinker.visual.`` prefix,
        strips it, and loads into ``self.visual``.
        """
        mapper = WeightsMapper(
            orig_to_new_prefix={"thinker.visual.": "visual."}
        )
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["thinker.audio_tower.", "thinker.model.", "thinker.lm_head.",
                           "talker.", "code2wav."],
        )
        loaded = loader.load_weights(weights, mapper=mapper)
        return loaded
