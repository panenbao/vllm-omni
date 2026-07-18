# SPDX-License-Identifier: Apache-2.0
"""Inter-stage payloads for the decoupled Qwen3-Omni encoder pipeline.

The audio and visual encoders are independently deployable, but the Thinker
still requires one embedding *per original multimodal item*.  We therefore
transport a typed ``embed.encoder`` list and its parallel
``meta.encoder_modalities`` list.  Consumers must reorder these items by the
original prompt placeholders; concatenating encoder output in stage order is
incorrect for prompts that interleave audio, images and video.
"""

from typing import Any

import torch
from vllm.inputs import TextPrompt

from vllm_omni.data_entry_keys import OmniPayload, OmniPayloadStruct, to_dict, to_struct
from vllm_omni.engine import OmniEngineCoreRequest
from vllm_omni.inputs.data import OmniTokensPrompt


_VALID_ENCODER_MODALITIES = frozenset({"audio", "image", "video"})


def _ensure_list(value: Any) -> Any:
    if hasattr(value, "_x"):
        return list(value._x)
    return list(value) if isinstance(value, tuple) else value


def _encoder_items(payload: dict[str, Any], *, legacy_modality: str | None = None) -> tuple[list[torch.Tensor], list[str]]:
    """Read the standard payload, accepting the former private key on input.

    The compatibility branch is deliberately read-only: all newly emitted
    payloads use the typed ``embed.encoder`` / ``meta.encoder_modalities``
    contract.
    """
    embed = payload.get("embed", {})
    meta = payload.get("meta", {})
    values = embed.get("encoder") if isinstance(embed, dict) else None
    labels = meta.get("encoder_modalities") if isinstance(meta, dict) else None
    # Pooling outputs are flattened by the AR/generation runners, while
    # connector payloads are nested.  Accept both representations at this
    # boundary and always emit the nested schema below.
    if values is None:
        values = payload.get("embed.encoder")
    if values is None:
        indexed = [
            (int(key.rsplit(".", 1)[1]), value)
            for key, value in payload.items()
            if key.startswith("embed.encoder.") and key.rsplit(".", 1)[1].isdigit()
        ]
        if indexed:
            values = [value for _, value in sorted(indexed)]
    if labels is None:
        labels = payload.get("meta.encoder_modalities")
    if values is None:
        values = payload.get("encoder_embeddings", [])
    if labels is None:
        labels = [legacy_modality] * len(values) if legacy_modality else []
    if isinstance(values, torch.Tensor):
        values = [values]
    if not isinstance(values, list) or not isinstance(labels, list) or len(values) != len(labels):
        return [], []
    items = [value for value in values if isinstance(value, torch.Tensor)]
    if len(items) != len(labels) or any(label not in _VALID_ENCODER_MODALITIES for label in labels):
        return [], []
    return items, list(labels)


def _pack_encoder_payload(pooling_output: dict[str, Any], *, legacy_modality: str | None) -> dict[str, Any]:
    embeddings, modalities = _encoder_items(pooling_output, legacy_modality=legacy_modality)
    payload: OmniPayload = {
        "embed": {"encoder": [embedding.detach().cpu() for embedding in embeddings]},
        "meta": {
            "encoder_modalities": modalities,
            "finished": torch.tensor(True, dtype=torch.bool),
        },
    }
    # Validate the outbound schema now, before it becomes an opaque SHM blob.
    return to_dict(to_struct(payload))


def audio2visual_full_payload(
    transfer_manager: Any,
    pooling_output: dict[str, Any],
    request: OmniEngineCoreRequest,
) -> dict[str, Any] | None:
    """Send audio encoder items to Stage 1 using the common payload schema."""
    if not isinstance(pooling_output, dict):
        return None
    return _pack_encoder_payload(pooling_output, legacy_modality="audio")


def visual2thinker_full_payload(
    transfer_manager: Any,
    pooling_output: dict[str, Any],
    request: OmniEngineCoreRequest,
) -> dict[str, Any] | None:
    """Send the carried audio plus new visual items to the Thinker stage."""
    if not isinstance(pooling_output, dict):
        return None
    return _pack_encoder_payload(pooling_output, legacy_modality=None)


def _token_only(source_outputs: list[Any]) -> list[OmniTokensPrompt]:
    # Full tensors are delivered through the connector into
    # model_intermediate_buffer.  The scheduler only needs prompt ids here.
    return [
        OmniTokensPrompt(
            prompt_token_ids=_ensure_list(output.prompt_token_ids),
            additional_information=None,
            multi_modal_data=None,
            mm_processor_kwargs=None,
        )
        for output in source_outputs
    ]


def audio2visual_token_only(
    source_outputs: list[Any],
    prompt: OmniTokensPrompt | TextPrompt | None = None,
    requires_multimodal_data: bool = False,
    streaming_context: Any | None = None,
) -> list[OmniTokensPrompt]:
    return _token_only(source_outputs)


def visual2thinker_token_only(
    source_outputs: list[Any],
    prompt: OmniTokensPrompt | TextPrompt | None = None,
    requires_multimodal_data: bool = False,
    streaming_context: Any | None = None,
) -> list[OmniTokensPrompt]:
    return _token_only(source_outputs)


def _prompt_with_encoder_payload(source_outputs: list[Any]) -> list[OmniTokensPrompt]:
    """Fallback direct path; mirrors the connector's standard payload."""
    result: list[OmniTokensPrompt] = []
    for output in source_outputs:
        completion = output.outputs[0]
        mm_output = getattr(completion, "multimodal_output", None)
        if not isinstance(mm_output, dict):
            continue
        payload = _pack_encoder_payload(mm_output, legacy_modality=None)
        result.append(
            OmniTokensPrompt(
                prompt_token_ids=_ensure_list(output.prompt_token_ids),
                additional_information=payload,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )
    return result


def audio2visual(
    source_outputs: list[Any],
    prompt: OmniTokensPrompt | TextPrompt | None = None,
    requires_multimodal_data: bool = False,
    streaming_context: Any | None = None,
) -> list[OmniTokensPrompt]:
    return _prompt_with_encoder_payload(source_outputs)


def visual2thinker(
    source_outputs: list[Any],
    prompt: OmniTokensPrompt | TextPrompt | None = None,
    requires_multimodal_data: bool = False,
    streaming_context: Any | None = None,
) -> list[OmniTokensPrompt]:
    return _prompt_with_encoder_payload(source_outputs)


def audio2visual_async_chunk(
    transfer_manager: Any,
    pooling_output: OmniPayload,
    request: OmniEngineCoreRequest,
    is_finished: bool = False,
) -> OmniPayloadStruct | None:
    payload = audio2visual_full_payload(transfer_manager, pooling_output, request)
    if payload is None:
        return None
    payload["meta"]["finished"] = torch.tensor(is_finished, dtype=torch.bool)
    return to_struct(payload)


def visual2thinker_async_chunk(
    transfer_manager: Any,
    pooling_output: OmniPayload,
    request: OmniEngineCoreRequest,
    is_finished: bool = False,
) -> OmniPayloadStruct | None:
    payload = visual2thinker_full_payload(transfer_manager, pooling_output, request)
    if payload is None:
        return None
    payload["meta"]["finished"] = torch.tensor(is_finished, dtype=torch.bool)
    return to_struct(payload)
