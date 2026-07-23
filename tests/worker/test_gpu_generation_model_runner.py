from types import SimpleNamespace

import pytest
import torch

from vllm_omni.worker.gpu_generation_model_runner import (
    ExecuteModelState,
    GPUGenerationModelRunner,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _DummyInputBatch:
    def __init__(self):
        self.req_ids = ["req-1"]
        self.req_id_to_index = {"req-1": 0}
        self.num_reqs = 1
        self.vocab_size = 10


class _DummyFeature:
    def __init__(self, modality, identifier, offset):
        self.modality = modality
        self.identifier = identifier
        self.mm_position = SimpleNamespace(offset=offset)


class _DummyRequestState:
    def __init__(self, mm_features):
        self.mm_features = mm_features


def _make_runner(multimodal_outputs):
    runner = object.__new__(GPUGenerationModelRunner)
    runner.execute_model_state = ExecuteModelState(
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        multimodal_outputs,
        None,
    )
    runner.kv_connector_output = None
    runner.input_batch = _DummyInputBatch()
    runner.use_async_scheduling = False
    runner.device = torch.device("cpu")
    runner.supports_mm_inputs = False
    runner.speculative_config = None
    runner.routed_experts_initialized = False
    return runner


def test_sample_tokens_tensor_output():
    multimodal_outputs = torch.randn(1, 2, 3)
    runner = _make_runner(multimodal_outputs)

    output = GPUGenerationModelRunner.sample_tokens(runner)

    assert len(output.pooler_output) == 1
    assert output.pooler_output[0]["model_outputs"].shape == (2, 3)


def test_sample_tokens_list_output():
    multimodal_outputs = [torch.randn(2, 1)]
    runner = _make_runner(multimodal_outputs)

    output = GPUGenerationModelRunner.sample_tokens(runner)

    assert len(output.pooler_output) == 1
    assert output.pooler_output[0]["model_outputs"].shape == (2, 1)


def test_sample_tokens_list_allows_none_output():
    multimodal_outputs = [None]
    runner = _make_runner(multimodal_outputs)

    output = GPUGenerationModelRunner.sample_tokens(runner)

    assert len(output.pooler_output) == 1
    assert output.pooler_output[0]["model_outputs"] is None


def test_sample_tokens_dict_output():
    multimodal_outputs = {"audio": torch.randn(1, 4), "unused": None}
    runner = _make_runner(multimodal_outputs)

    output = GPUGenerationModelRunner.sample_tokens(runner)

    assert len(output.pooler_output) == 1
    assert "audio" in output.pooler_output[0]
    assert "unused" not in output.pooler_output[0]
    assert output.pooler_output[0]["audio"].shape == (1, 4)


def test_build_encoder_outputs_preserves_request_and_prompt_order():
    """Mixed multimodal batches must not use a batch-global modality queue."""
    runner = object.__new__(GPUGenerationModelRunner)
    runner.model = SimpleNamespace(model_stage="visual_encoder")
    runner.input_batch = SimpleNamespace(req_ids=["req-1", "req-2"])
    runner.requests = {
        "req-1": _DummyRequestState(
            [
                _DummyFeature("video", "video-1", 10),
                _DummyFeature("audio", "audio-1", 20),
                _DummyFeature("image", "image-1", 30),
            ]
        ),
        "req-2": _DummyRequestState(
            [
                _DummyFeature("audio", "audio-2", 10),
                _DummyFeature("video", "video-2", 20),
            ]
        ),
    }
    runner.encoder_cache = {
        "video-1": torch.tensor([[11.0]]),
        "image-1": torch.tensor([[12.0]]),
        "video-2": torch.tensor([[21.0]]),
    }
    runner.model_intermediate_buffer = {
        "req-1": {
            "embed": {"encoder": [torch.tensor([[13.0]])]},
            "meta": {"encoder_modalities": ["audio"]},
        },
        "req-2": {
            "embed": {"encoder": [torch.tensor([[22.0]])]},
            "meta": {"encoder_modalities": ["audio"]},
        },
    }

    outputs = GPUGenerationModelRunner._build_decoupled_encoder_outputs(runner)

    assert [output["meta"]["encoder_modalities"] for output in outputs] == [
        ["video", "audio", "image"],
        ["audio", "video"],
    ]
    assert [item.item() for item in outputs[0]["embed"]["encoder"]] == [11.0, 13.0, 12.0]
    assert [item.item() for item in outputs[1]["embed"]["encoder"]] == [22.0, 21.0]


def test_build_encoder_outputs_handles_multiple_items_in_one_request():
    runner = object.__new__(GPUGenerationModelRunner)
    runner.model = SimpleNamespace(model_stage="audio_encoder")
    runner.input_batch = SimpleNamespace(req_ids=["req-1"])
    runner.requests = {
        "req-1": _DummyRequestState(
            [
                _DummyFeature("audio", "audio-2", 20),
                _DummyFeature("audio", "audio-1", 10),
            ]
        )
    }
    runner.encoder_cache = {
        "audio-1": torch.tensor([[1.0]]),
        "audio-2": torch.tensor([[2.0]]),
    }
    runner.model_intermediate_buffer = {}

    outputs = GPUGenerationModelRunner._build_decoupled_encoder_outputs(runner)

    assert outputs[0]["meta"]["encoder_modalities"] == ["audio", "audio"]
    assert [item.item() for item in outputs[0]["embed"]["encoder"]] == [1.0, 2.0]
