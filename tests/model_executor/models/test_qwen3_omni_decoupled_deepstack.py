from types import SimpleNamespace

import torch
import torch.nn as nn

from vllm_omni.model_executor.models.qwen3_omni_decoupled.qwen3_omni_llm import (
    Qwen3OmniMoeLLMStage,
)


class _FakeLanguageModel(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return torch.full(
            (input_ids.numel(), self.hidden_size),
            -1.0,
            dtype=torch.float32,
            device=input_ids.device,
        )


def _make_stage(*, hidden_size: int = 2, num_levels: int = 2, capacity: int = 16):
    stage = Qwen3OmniMoeLLMStage.__new__(Qwen3OmniMoeLLMStage)
    nn.Module.__init__(stage)
    stage.config = SimpleNamespace(
        text_config=SimpleNamespace(hidden_size=hidden_size),
        vision_config=SimpleNamespace(
            deepstack_visual_indexes=list(range(num_levels))
        ),
        image_token_id=20,
        video_token_id=21,
        audio_token_id=30,
    )
    stage.language_model = _FakeLanguageModel(hidden_size)
    stage._ds_idx = list(range(num_levels))
    stage._ds_num_level = num_levels
    stage._ds_input_embeds = [
        torch.zeros(capacity, hidden_size) for _ in range(num_levels)
    ]
    stage._ds_input_embeds_num_tokens = 0
    return stage


def test_deepstack_is_scattered_only_to_visual_token_positions():
    stage = _make_stage()
    input_ids = torch.tensor([1, 2, 20, 20, 3, 30, 4, 5])
    is_multimodal = torch.tensor(
        [False, False, True, True, False, True, False, False]
    )

    # Layout per visual row: [main | deepstack level 0 | deepstack level 1].
    visual = torch.tensor(
        [
            [10.0, 11.0, 20.0, 21.0, 30.0, 31.0],
            [12.0, 13.0, 22.0, 23.0, 32.0, 33.0],
        ]
    )
    audio = torch.tensor([[40.0, 41.0]])

    merged = stage.embed_input_ids(
        input_ids,
        multimodal_embeddings=[visual, audio],
        is_multimodal=is_multimodal,
    )

    torch.testing.assert_close(merged[2:4], visual[:, :2])
    torch.testing.assert_close(merged[5], audio[0])

    expected_level_0 = torch.zeros(8, 2)
    expected_level_0[2:4] = visual[:, 2:4]
    expected_level_1 = torch.zeros(8, 2)
    expected_level_1[2:4] = visual[:, 4:6]
    torch.testing.assert_close(stage._ds_input_embeds[0][:8], expected_level_0)
    torch.testing.assert_close(stage._ds_input_embeds[1][:8], expected_level_1)
    assert stage._ds_input_embeds_num_tokens == input_ids.numel()


def test_deepstack_clear_removes_prefill_features_before_decode():
    stage = _make_stage(capacity=8)
    deepstack = torch.arange(2 * 6 * 2, dtype=torch.float32).reshape(2, 6, 2)
    stage._ds_set_input_embeds(deepstack)

    # Forward can be padded beyond the unpadded prefill length.
    padded = stage._ds_get_input_embeds(8)
    assert padded is not None
    torch.testing.assert_close(
        padded["deepstack_input_embeds_0"][6:8], torch.zeros(2, 2)
    )

    stage._ds_clear(8)

    assert stage._ds_input_embeds_num_tokens == 0
    torch.testing.assert_close(stage._ds_input_embeds[0], torch.zeros(8, 2))
    torch.testing.assert_close(stage._ds_input_embeds[1], torch.zeros(8, 2))
