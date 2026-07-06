# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage input processors for decoupled Qwen3-Omni-MoE (5-stage pipeline).

Flow: audio_encoder(0) → visual_encoder(1) → thinker_lm(2) → talker(3) → code2wav(4)

Transitions:
  audio_encoder(0) → visual_encoder(1):
    audio embeddings passed via shared memory connector
  visual_encoder(1) → thinker_lm(2):
    combined audio + visual embeddings → LLM input
  thinker_lm(2) → talker(3):
    reuses existing qwen3_omni.thinker2talker functions
  talker(3) → code2wav(4):
    reuses existing qwen3_omni.talker2code2wav functions
"""

import logging
from typing import Any

import torch
from vllm.inputs import TextPrompt

from vllm_omni.data_entry_keys import (
    OmniPayload,
    OmniPayloadStruct,
)
from vllm_omni.engine import OmniEngineCoreRequest
from vllm_omni.inputs.data import OmniTokensPrompt

logger = logging.getLogger(__name__)


def _ensure_list(x):
    """Convert ConstantList / tensor-like to Python list."""
    if hasattr(x, "_x"):
        return list(x._x)
    elif not isinstance(x, list):
        return x
    return list(x)


# =========================
# Thinker LM Stage Input (from upstream encoders)
# =========================


def encoder2thinker_lm(
    source_outputs: list[Any],
    prompt: OmniTokensPrompt | TextPrompt | None = None,
    requires_multimodal_data: bool = False,
    streaming_context: Any | None = None,
) -> list[OmniTokensPrompt]:
    """Orchestrator-side input processor for thinker_lm (stage 2).

    Receives the output of visual_encoder (stage 1), which contains combined
    audio + visual embeddings accumulated from stages 0 and 1.
    """
    encoder_outputs = source_outputs
    lm_inputs: list[OmniTokensPrompt] = []

    for i, enc_output in enumerate(encoder_outputs):
        output = enc_output.outputs[0]
        req_id = str(getattr(enc_output, "request_id", f"idx-{i}"))
        mm_raw = getattr(output, "multimodal_output", None)
        if not isinstance(mm_raw, dict):
            logger.debug("encoder2thinker_lm: skip req=%s due to empty multimodal_output", req_id)
            continue

        encoder_embeddings = mm_raw.get("encoder_embeddings", [])
        if not encoder_embeddings:
            logger.debug("encoder2thinker_lm: skip req=%s due to missing encoder_embeddings", req_id)
            continue

        prompt_token_ids = _ensure_list(enc_output.prompt_token_ids)

        info: dict[str, Any] = {
            "encoder_embeddings": encoder_embeddings,
            "prompt_token_ids": prompt_token_ids,
        }

        lm_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=prompt_token_ids,
                additional_information=info if info else None,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )

    return lm_inputs


def encoder2thinker_lm_token_only(
    source_outputs: list[Any],
    prompt: OmniTokensPrompt | TextPrompt | None = None,
    requires_multimodal_data: bool = False,
    streaming_context: Any | None = None,
) -> list[OmniTokensPrompt]:
    """Sync connector variant for encoder→thinker_lm transition.

    Bulk encoder embeddings arrive via the connector's shared memory path.
    This function only produces the placeholder prompt for KV-cache allocation.
    """
    lm_inputs: list[OmniTokensPrompt] = []
    for i, enc_output in enumerate(source_outputs):
        prompt_token_ids = _ensure_list(enc_output.prompt_token_ids)
        lm_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=prompt_token_ids,
                additional_information=None,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )
    return lm_inputs


# =========================
# Connector Path (full_payload / async_chunk)
# =========================


def audio2visual_full_payload(
    transfer_manager: Any,
    pooling_output: dict[str, Any],
    request: OmniEngineCoreRequest,
) -> dict[str, Any] | None:
    """Connector next-stage for audio_encoder (stage 0) → visual_encoder (stage 1).

    Packages audio encoder embeddings for shared memory transfer.
    """
    rid = getattr(request, "request_id", None)
    if not isinstance(pooling_output, dict):
        logger.warning(
            "audio2visual_full_payload: pooling_output not a dict (type=%s) for req=%s",
            type(pooling_output).__name__, rid,
        )
        return None

    encoder_embeddings = pooling_output.get("encoder_embeddings")
    if encoder_embeddings is None:
        logger.debug("audio2visual_full_payload: no encoder_embeddings for req=%s", rid)
        return None

    if isinstance(encoder_embeddings, torch.Tensor):
        encoder_embeddings = [encoder_embeddings]

    return {
        "encoder_embeddings": [e.detach().cpu() if isinstance(e, torch.Tensor) else e for e in encoder_embeddings],
        "meta": {"finished": torch.tensor(True, dtype=torch.bool)},
    }


def visual2thinker_full_payload(
    transfer_manager: Any,
    pooling_output: dict[str, Any],
    request: OmniEngineCoreRequest,
) -> dict[str, Any] | None:
    """Connector next-stage for visual_encoder (stage 1) → thinker_lm (stage 2).

    Packages combined audio + visual embeddings for shared memory transfer.
    """
    rid = getattr(request, "request_id", None)
    if not isinstance(pooling_output, dict):
        logger.warning(
            "visual2thinker_full_payload: pooling_output not a dict (type=%s) for req=%s",
            type(pooling_output).__name__, rid,
        )
        return None

    encoder_embeddings = pooling_output.get("encoder_embeddings")
    if encoder_embeddings is None:
        logger.debug("visual2thinker_full_payload: no encoder_embeddings for req=%s", rid)
        return None

    if isinstance(encoder_embeddings, torch.Tensor):
        encoder_embeddings = [encoder_embeddings]

    return {
        "encoder_embeddings": [e.detach().cpu() if isinstance(e, torch.Tensor) else e for e in encoder_embeddings],
        "meta": {"finished": torch.tensor(True, dtype=torch.bool)},
    }


def audio2visual_async_chunk(
    transfer_manager: Any,
    pooling_output: OmniPayload,
    request: OmniEngineCoreRequest,
    is_finished: bool = False,
) -> OmniPayloadStruct | None:
    """Async chunk variant for audio_encoder → visual_encoder."""
    payload = audio2visual_full_payload(transfer_manager, pooling_output, request)
    if payload is None:
        return None
    payload["meta"]["finished"] = torch.tensor(is_finished, dtype=torch.bool)
    return payload


def visual2thinker_async_chunk(
    transfer_manager: Any,
    pooling_output: OmniPayload,
    request: OmniEngineCoreRequest,
    is_finished: bool = False,
) -> OmniPayloadStruct | None:
    """Async chunk variant for visual_encoder → thinker_lm."""
    payload = visual2thinker_full_payload(transfer_manager, pooling_output, request)
    if payload is None:
        return None
    payload["meta"]["finished"] = torch.tensor(is_finished, dtype=torch.bool)
    return payload
