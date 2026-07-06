# Omni Model Directory Structure

## Required Files

```
vllm_omni/model_executor/models/
└── your_model_name/
    ├── __init__.py                    # Exports the main unified model class
    ├── your_model.py                  # Unified orchestrator: dispatches by model_stage
    ├── your_model_thinker.py          # Thinker stage: multimodal understanding → text
    ├── your_model_talker.py           # Talker stage: text embeddings → codec codes
    ├── your_model_code2wav.py         # Code2Wav stage: RVQ codes → audio waveform
    └── configuration_your_model.py    # HF-style config classes (optional)

vllm_omni/model_executor/stage_input_processors/
└── your_model_name.py                 # Stage transition functions (thinker→talker, talker→code2wav)

vllm_omni/deploy/
└── your_model.yaml                    # Stage deploy configuration
```

## Registration

```
vllm_omni/model_executor/models/registry.py     # Add entries to _OMNI_MODELS dict
```

## Testing

```
tests/e2e/offline_inference/test_your_model.py   # E2E offline tests
tests/e2e/online_serving/test_your_model.py       # E2E online serving tests (optional)
```

## Examples

```
examples/offline_inference/your_model_name/
├── end2end.py                         # Offline inference example
└── README.md                          # Usage instructions

examples/online_serving/your_model_name/
├── speech_client.py                   # API client example
├── run_server.sh                      # Server launcher
└── gradio_demo.py                     # Interactive demo (optional)
```

## File Purposes

### `your_model.py` (Unified Orchestrator)
- Reads `model_stage` from `vllm_config.model_config.model_stage`
- Initializes the correct stage: `thinker`, `talker`, `code2wav`
- Sets `self.model = self.<stage>` for vLLM's dispatch
- Implements `load_weights()` with prefix routing
- Implements `forward()`, `compute_logits()`, `sample()`

### `your_model_thinker.py`
- Inherits from a vLLM base model (e.g., Qwen3MoeForCausalLM)
- Implements `SupportsMultiModal` for multimodal input processing
- Captures intermediate hidden states in `multimodal_outputs`
- Audio/video/image encoder initialization

### `your_model_talker.py`
- Inherits from same vLLM base as thinker (weight reuse)
- Replaces LM head with codec head (multi-codebook prediction)
- Receives thinker embeddings via input processor
- Outputs 2D codec codes tensor [num_codebooks, seq_len]

### `your_model_code2wav.py`
- Pure nn.Module (not a vLLM causal LM)
- Contains audio decoder / vocoder
- Handles chunked decoding for streaming
- Outputs audio waveform tensor

### `stage_input_processors/your_model_name.py`
- Thinker→Talker: extracts embeddings/hidden states, builds OmniPayload
- Talker→Code2Wav: flattens codec codes into token ID sequence
- Async chunk variants for realtime streaming
- PD disaggregation support (prefill+decode embedding merge)
