import torch

from vllm_omni.data_entry_keys import deserialize_payload, serialize_payload
from vllm_omni.model_executor.stage_input_processors.qwen3_omni_decoupled import (
    audio2visual_full_payload,
    visual2thinker_full_payload,
)


class _Request:
    request_id = "decoupled-contract-test"


def test_encoder_payload_preserves_item_boundaries_and_modalities():
    """Audio and visual tensors must never be concatenated at the boundary."""
    audio = torch.ones(3, 4, dtype=torch.bfloat16)
    image = torch.ones(2, 4, dtype=torch.bfloat16)
    request = _Request()

    audio_payload = audio2visual_full_payload(
        None,
        {"embed": {"encoder": [audio]}, "meta": {"encoder_modalities": ["audio"]}},
        request,
    )
    assert audio_payload is not None
    assert audio_payload["meta"]["encoder_modalities"] == ["audio"]
    assert [item.shape for item in audio_payload["embed"]["encoder"]] == [(3, 4)]
    assert audio_payload["embed"]["encoder"][0].device.type == "cpu"

    thinker_payload = visual2thinker_full_payload(
        None,
        {
            "embed": {"encoder": [audio, image]},
            "meta": {"encoder_modalities": ["audio", "image"]},
        },
        request,
    )
    assert thinker_payload is not None
    assert thinker_payload["meta"]["encoder_modalities"] == ["audio", "image"]
    assert [item.shape for item in thinker_payload["embed"]["encoder"]] == [(3, 4), (2, 4)]


def test_encoder_payload_accepts_flattened_pooler_output():
    """Generation runners flatten nested output before full-payload packing."""
    video = torch.ones(5, 4, dtype=torch.bfloat16)
    payload = visual2thinker_full_payload(
        None,
        {"embed.encoder": [video], "meta.encoder_modalities": ["video"]},
        _Request(),
    )
    assert payload is not None
    assert payload["meta"]["encoder_modalities"] == ["video"]
    assert payload["embed"]["encoder"][0].shape == (5, 4)


def test_additional_information_round_trip_preserves_encoder_tensor_list():
    """The non-connector path must not encode encoder tensors as list_data."""
    payload = {
        "embed": {"encoder": [torch.ones(2, 4), torch.ones(3, 4)]},
        "meta": {"encoder_modalities": ["audio", "video"]},
    }
    wire = serialize_payload(payload)
    assert wire is not None
    assert "embed.encoder.0" in wire.entries
    restored = deserialize_payload(wire)
    assert [tensor.shape for tensor in restored["embed"]["encoder"]] == [(2, 4), (3, 4)]
    assert restored["meta"]["encoder_modalities"] == ["audio", "video"]
