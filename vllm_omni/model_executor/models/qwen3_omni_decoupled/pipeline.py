# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pipeline topology for decoupled Qwen3-Omni-MoE (5-stage).

Stage 0: Audio Encoder   — audio → audio embeddings
Stage 1: Visual Encoder  — image/video → visual embeddings
Stage 2: Thinker LM      — text + embeddings → hidden states (AR generation)
Stage 3: Talker          — hidden states → RVQ codec codes
Stage 4: Code2Wav        — RVQ codes → audio waveform

Sequential routing: 0 → 1 → 2 → 3 → 4
Encoder stages pass through when no matching input is present.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

_DECOUPLED_PROC = "vllm_omni.model_executor.stage_input_processors.qwen3_omni_decoupled"
_SHARED_PROC = "vllm_omni.model_executor.stage_input_processors.qwen3_omni"

QWEN3_OMNI_DECOUPLED_PIPELINE = PipelineConfig(
    model_type="qwen3_omni_moe_decoupled",
    model_arch="Qwen3OmniMoeDecoupledForConditionalGeneration",
    hf_architectures=("Qwen3OmniMoeForConditionalGeneration",),
    stages=(
        # ── Stage 0: Audio Encoder ──────────────────────────────────
        StagePipelineConfig(
            stage_id=0,
            model_stage="audio_encoder",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(),
            requires_multimodal_data=True,
            hf_config_name="thinker_config",
            engine_output_type="latent",
            custom_process_next_stage_input_func=(f"{_DECOUPLED_PROC}.audio2visual_full_payload"),
            async_chunk_process_next_stage_input_func=(f"{_DECOUPLED_PROC}.audio2visual_async_chunk"),
        ),
        # ── Stage 1: Visual Encoder ─────────────────────────────────
        StagePipelineConfig(
            stage_id=1,
            model_stage="visual_encoder",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(0,),
            requires_multimodal_data=True,
            hf_config_name="thinker_config",
            engine_output_type="latent",
            custom_process_input_func=f"{_DECOUPLED_PROC}.audio2visual",
            sync_process_input_func=f"{_DECOUPLED_PROC}.audio2visual_token_only",
            custom_process_next_stage_input_func=(f"{_DECOUPLED_PROC}.visual2thinker_full_payload"),
            async_chunk_process_next_stage_input_func=(f"{_DECOUPLED_PROC}.visual2thinker_async_chunk"),
        ),
        # ── Stage 2: Thinker LM ─────────────────────────────────────
        StagePipelineConfig(
            stage_id=2,
            model_stage="thinker_lm",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(1,),
            final_output=True,
            final_output_type="text",
            owns_tokenizer=True,
            hf_config_name="thinker_config",
            engine_output_type="latent",
            custom_process_input_func=f"{_DECOUPLED_PROC}.visual2thinker",
            sync_process_input_func=f"{_DECOUPLED_PROC}.visual2thinker_token_only",
            custom_process_next_stage_input_func=(f"{_SHARED_PROC}.thinker2talker_full_payload"),
            async_chunk_process_next_stage_input_func=(f"{_SHARED_PROC}.thinker2talker_async_chunk"),
            sampling_constraints={"detokenize": True},
        ),
        # ── Stage 3: Talker ─────────────────────────────────────────
        StagePipelineConfig(
            stage_id=3,
            model_stage="talker",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(2,),
            hf_config_name="talker_config",
            engine_output_type="latent",
            custom_process_input_func=f"{_SHARED_PROC}.thinker2talker",
            sync_process_input_func=f"{_SHARED_PROC}.thinker2talker_token_only",
            custom_process_next_stage_input_func=(f"{_SHARED_PROC}.talker2code2wav_full_payload"),
            async_chunk_process_next_stage_input_func=(f"{_SHARED_PROC}.talker2code2wav_async_chunk"),
            sampling_constraints={
                "detokenize": False,
                "stop_token_ids": [2150],
            },
        ),
        # ── Stage 4: Code2Wav ───────────────────────────────────────
        StagePipelineConfig(
            stage_id=4,
            model_stage="code2wav",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(3,),
            final_output=True,
            final_output_type="audio",
            hf_config_name="thinker_config",
            engine_output_type="audio",
            custom_process_input_func=f"{_SHARED_PROC}.talker2code2wav",
            sampling_constraints={"detokenize": True},
        ),
    ),
)
