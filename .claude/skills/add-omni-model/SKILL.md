---
name: add-omni-model
description: "Add a new omni-modality model (multi-stage: thinker/talker/code2wav) to vLLM-Omni from HuggingFace reference through production serving. Use when integrating a new multi-stage omni model, wiring stage separation for vision+audio+text understanding/generation, creating stage input processors, or enabling multi-modal pipeline serving."
---

# Adding an Omni-Modality Model to vLLM-Omni

## Overview

vLLM-Omni supports **multi-stage model architectures** where different stages can run on different devices and process different modalities. Each stage is independently configurable and can target separate GPUs.

### Architecture Patterns

Three common patterns (Qwen3-Omni is the canonical three-stage reference):

| Pattern | Stages | Example |
|---------|--------|---------|
| **Three-stage** | Thinker (MM understanding) → Talker (codec) → Code2Wav (audio) | Qwen3-Omni, Qwen2.5-Omni |
| **Two-stage TTS** | Talker (AR code predictor) → Code2Wav (audio decoder) | Qwen3-TTS, CosyVoice3, Fish Speech |
| **Single-stage AR** | Bundled talker+code2wav in one worker, streaming via generator | MOSS-TTS-Nano |

This skill focuses on the **omni-modality pattern** (thinker + optional downstream stages), where the thinker stage handles text + audio + video inputs.

## Cross-Cutting Invariants

These rules apply to every omni model. Check them at every phase.

### I1. Stage separation boundary

Each stage is a **separate model class** in a separate file. Stages communicate via `OmniOutput` / `OmniTokensPrompt`. Never import stage A's internals directly in stage B.

### I2. OmniOutput contract

Model `forward()` must return a `NamedTuple`:
```python
class OmniOutput(NamedTuple):
    text_hidden_states: torch.Tensor
    multimodal_outputs: OmniPayload | None = None
    intermediate_tensors: IntermediateTensors | None = None
    next_token_id: torch.Tensor | None = None
```

Keys in `multimodal_outputs` must be strings. Store intermediate embeddings/hidden states with meaningful keys (e.g., `"0"` for embedding layer, `"24"` for projection layer, `"codes"` for codec codes).

### I3. Hot-loop GPU discipline

Inside any per-step model loop (AR decode, audio chunk loop):
- No `tensor.item()`, `.cpu()`, `.tolist()` — each triggers GPU→CPU sync
- No Python-side control flow depending on tensor values; use `torch.where` / masking
- No per-step `torch.cat` in hot path; pre-allocate buffers

### I4. Multimodal output consumer hygiene

`outputs[0].outputs[0].multimodal_output[<key>]` can be `Tensor`, `list[Tensor]`, `np.ndarray`, or scalar. Never use `dict.get("a") or dict.get("b")` on tensor values. Always handle list form: `if isinstance(x, list): x = torch.cat([t.reshape(-1) for t in x], dim=0)`.

### I5. Weight prefix discipline

Omni model weights use component prefixes (`thinker.`, `talker.`, `code2wav.`). The unified model's `load_weights()` must:
1. Separate weights by prefix
2. Strip prefix before dispatching to sub-model
3. Add prefix back to returned set of loaded weight names

### I6. Stage config YAML completeness

Every stage entry in the deploy YAML must specify: `stage_id`, `max_num_seqs`, `gpu_memory_utilization`, `trust_remote_code`, `devices`. For the thinker stage, `enable_prefix_caching: false` is typical. For downstream stages, configure `input_connectors`.

### I7. Stage input processor: prompt token length

The talker stage prompt must be exactly the right length for KV-cache allocation. Use `_compute_talker_prompt_ids_length()`-style logic to count user+assistant tokens — don't hardcode.

---

## Phase 1: Directory Structure & Config Classes

```text
vllm_omni/model_executor/models/
└── your_model_name/
    ├── __init__.py                   # Export main class
    ├── your_model.py                 # Unified orchestrator class
    ├── your_model_thinker.py         # Thinker stage (MM → text)
    ├── your_model_talker.py          # Talker stage (text → codes) [optional]
    ├── your_model_code2wav.py        # Code2Wav stage (codes → audio) [optional]
    └── ... (other stage files)

vllm_omni/model_executor/stage_input_processors/
└── your_model_name.py                # Stage transition functions

vllm_omni/deploy/
└── your_model.yaml                   # Stage deploy config
```

### Tasks

1. **Create model directory** under `vllm_omni/model_executor/models/`
2. **Create config classes** (`configuration_<model>.py`) if needed, with `model_type` registration
3. **Understand the reference** — run the HF model, document architecture, tokenizer config, sub-models
4. **Document key constants**: special token IDs, codebook size, hop length, sample rate

### Key Questions

- How many stages? What does each do?
- What is the token vocabulary? (special tokens, codec offsets, modality tokens)
- What multimodal inputs? (audio, video, image — what encoders are needed?)
- How are codec codes structured? (num codebooks, codebook size, RVQ layers)
- What are the hidden-state layer indices needed for talker conditioning?

---

## Phase 2: Implement Stage Components

### 2.1 Thinker Stage

The thinker handles multimodal understanding. Inherit from a suitable base model in vLLM:

```python
from vllm.model_executor.models.interfaces import SupportsMultiModal, SupportsPP

class YourThinkerForConditionalGeneration(
    BaseModel, SupportsMultiModal, SupportsPP
):
    """Thinker stage: multimodal understanding → text generation."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # Initialize base model
        # Set up multimodal processors (audio/video/image encoders)
        # Register hooks for embedding capture
        pass

    def forward(self, ...):
        # Process multimodal inputs
        # Generate text + capture intermediate hidden states
        return OmniOutput(
            text_hidden_states=hidden_states,
            multimodal_outputs={
                "0": captured_embeddings,       # word embedding layer
                "24": captured_hidden_states,   # projection layer
                "tts_bos": tts_bos_embed,
                "tts_eos": tts_eos_embed,
            },
        )
```

Key interfaces to implement:
- `SupportsMultiModal` — if processing images/audio/video
- `SupportsPP` — for pipeline parallelism
- `SupportsMRoPE` — for multi-dimensional RoPE (if applicable)

Register with the multimodal registry if needed:
```python
@MULTIMODAL_REGISTRY.register_processor(
    YourMultiModalProcessor,
    info=YourProcessingInfo,
    dummy_inputs=YourDummyInputsBuilder,
)
```

### 2.2 Talker Stage (optional)

Converts text embeddings to codec codes. Inherits from the same base but replaces LM head:

```python
class YourTalkerForConditionalGeneration(BaseModel, SupportsPP):
    """Talker stage: text embeddings → RVQ codec codes."""

    def __init__(self, vllm_config, talker_config, prefix):
        # Initialize base
        # Replace LM head with codec head (multi-codebook predictor)
        # Set up text projection from thinker embeddings
        pass

    def forward(self, ...):
        # Project thinker embeddings → codec logits
        # Return OmniOutput with multimodal_outputs["codes"]["audio"]
        ...
```

### 2.3 Code2Wav Stage (optional)

Generates audio waveform from RVQ codes:

```python
class YourCode2Wav(nn.Module):
    """Code2Wav stage: RVQ codes → audio waveform."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # Initialize audio decoder / vocoder
        # Set up codec processing
        pass
```

---

## Phase 3: Unified Model Class

The main model class (`your_model.py`) orchestrates all stages:

```python
@MULTIMODAL_REGISTRY.register_processor(...)
class YourModelForConditionalGeneration(
    nn.Module, SupportsMultiModal, SupportsPP, YourMixin
):
    """Unified model combining thinker, talker, and code2wav."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.have_multimodal_outputs = True
        config = vllm_config.model_config.hf_config
        self.model_stage = vllm_config.model_config.model_stage

        if self.model_stage == "thinker":
            thinker_config = vllm_config.with_hf_config(
                config.thinker_config,
                architectures=["YourThinkerForConditionalGeneration"]
            )
            self.thinker = init_vllm_registered_model(
                vllm_config=thinker_config,
                prefix=maybe_prefix(prefix, "thinker"),
                hf_config=config.thinker_config,
                architectures=["YourThinkerForConditionalGeneration"],
            )
            self.model = self.thinker

        elif self.model_stage == "talker":
            # Similar pattern for talker...
            pass

        elif self.model_stage == "code2wav":
            # Similar pattern for code2wav...
            pass
        else:
            raise ValueError(...)

    def forward(self, ...): ...
    def load_weights(self, weights): ...
    def compute_logits(self, hidden_states, sampling_metadata): ...
    def sample(self, ...): ...
```

### Key Methods

| Method | Purpose |
|--------|---------|
| `forward()` | Dispatch to stage's forward, capture hidden states |
| `load_weights()` | Separate weights by prefix, dispatch to sub-models |
| `compute_logits()` | Compute output logits from hidden states |
| `sample()` | Sample tokens from logits |
| `embed_input_ids()` | Embed input token IDs |
| `embed_multimodal()` | Process multimodal inputs (thinker only) |

---

## Phase 4: Model Registration

Register in `vllm_omni/model_executor/models/registry.py`:

```python
_OMNI_MODELS = {
    # ... existing models ...

    "YourModelForConditionalGeneration": (
        "your_model_name",                    # Module folder
        "your_model",                         # Module file (no .py)
        "YourModelForConditionalGeneration",  # Class name
    ),
    "YourModelThinkerForConditionalGeneration": (
        "your_model_name",
        "your_model_thinker",
        "YourModelThinkerForConditionalGeneration",
    ),
    # ... other stages if separate classes ...
}
```

Each entry is `(folder_name, module_file_name, class_name)`. The registry uses lazy loading.

Create `__init__.py` exporting the main class:
```python
from .your_model import YourModelForConditionalGeneration
__all__ = ["YourModelForConditionalGeneration"]
```

---

## Phase 5: Stage Configuration YAML

See `vllm_omni/deploy/qwen3_omni_moe.yaml` for the reference format.

```yaml
async_chunk: true

connectors:
  connector_of_shared_memory:
    name: SharedMemoryConnector
    extra:
      initial_codec_chunk_frames: 4
      codec_chunk_frames: 25
      codec_left_context_frames: 25

stages:
  - stage_id: 0            # Thinker
    max_num_seqs: 64
    gpu_memory_utilization: 0.9
    trust_remote_code: true
    enable_prefix_caching: false
    devices: "0,1,2,3"
    tensor_parallel_size: 4
    default_sampling_params:
      temperature: 0.0
      max_tokens: 2048

  - stage_id: 1            # Talker
    max_num_seqs: 64
    gpu_memory_utilization: 0.6
    trust_remote_code: true
    devices: "4"
    input_connectors:
      from_stage_0: connector_of_shared_memory
    default_sampling_params:
      temperature: 0.9
      max_tokens: 4096

  - stage_id: 2            # Code2Wav
    max_num_batched_tokens: 65536
    gpu_memory_utilization: 0.1
    enforce_eager: false
    trust_remote_code: true
    devices: "4"
    input_connectors:
      from_stage_1: connector_of_shared_memory

platforms:
  npu:
    stages:
      - stage_id: 0
        gpu_memory_utilization: 0.6
        tensor_parallel_size: 2
        devices: "0,1"
      # ... per-platform overrides ...
```

Key configuration fields:

| Field | Description |
|-------|-------------|
| `model_stage` | Which stage ("thinker", "talker", "code2wav") |
| `model_arch` | Architecture name (must match registry key) |
| `engine_input_source` | List of stage IDs providing input |
| `custom_process_input_func` | Function for stage transition processing |
| `final_output` | Whether this is the final stage (True/False) |
| `final_output_type` | "text", "audio", "image", etc. |
| `async_chunk` | Enable inter-stage streaming (top-level field) |
| `devices` | GPU device assignment |
| `tensor_parallel_size` | TP degree for this stage |
| `input_connectors` | Shared memory configuration between stages |

### Platform-specific overrides

Use `platforms:` section (same structure as `stages:`) to override per-stage config for specific hardware:

Available platforms: `npu`, `rocm`, `xpu`. Fields not defined fall back to the top-level `stages:` values.

---

## Phase 6: Stage Input Processors

Create `vllm_omni/model_executor/stage_input_processors/your_model.py`.

### Thinker → Talker

```python
def thinker2talker(
    source_outputs: list[Any],
    prompt: OmniTokensPrompt | TextPrompt | None = None,
    requires_multimodal_data: bool = False,
    streaming_context: Any | None = None,
) -> list[OmniTokensPrompt]:
    """
    1. Extract thinker embeddings + hidden states from multimodal_output
    2. Build OmniPayload with embed/hidden_states/ids
    3. Compute talker prompt length
    4. Return OmniTokensPrompt with [0]*prompt_len + payload as additional_information
    """
    ...
```

### Talker → Code2Wav

```python
def talker2code2wav(
    source_outputs: list[Any],
    prompt: OmniTokensPrompt | TextPrompt | None = None,
    requires_multimodal_data: bool = False,
    streaming_context: Any | None = None,
) -> list[OmniTokensPrompt]:
    """
    1. Extract codec codes from multimodal_output["codes"]["audio"]
    2. Transpose, flatten, convert to list[int]
    3. Return OmniTokensPrompt with codec_codes as prompt_token_ids
    """
    ...
```

### Data Flow

```
Thinker forward()
  → OmniOutput(text_hidden_states, multimodal_outputs={"0": emb, "24": hid, ...})
  → orchestrator stores via stage_client.set_engine_outputs()

Stage input processor (thinker2talker)
  → reads stage_list[source_stage_id].engine_outputs
  → builds OmniPayload with embed/prefill/tts_bos/tts_eos/ids/meta
  → returns list[OmniTokensPrompt]

Talker receives OmniTokensPrompt with additional_information containing OmniPayload
```

### Key Data Structures

**Input to your processor:**
- `stage_list[source_stage_id].engine_outputs`: list of `EngineCoreOutput`
  - Each has `.outputs`: list of `RequestOutput`
  - Each `RequestOutput` has `.multimodal_output`: dict with model-specific keys

**Output from your processor:**
- Must return `list[OmniTokensPrompt]` where each has:
  - `prompt_token_ids`: list[int] — token IDs for the next stage
  - `additional_information`: dict — metadata (embeddings, hidden states, etc.)
  - `multi_modal_data`: optional multimodal data

### Shared Memory Connector (Async Chunk Mode)

For realtime streaming with `async_chunk: true`, the connector path uses a different function signature:

```python
def thinker2talker_async_chunk(
    transfer_manager: Any,
    pooling_output: OmniPayload,
    request: OmniEngineCoreRequest,
    is_finished: bool = False,
) -> OmniPayloadStruct | None:
```

The connector handles chunking: first chunk caches prefill embeddings and returns None (no downstream emission), subsequent decode chunks flush the cache.

### PD Disaggregation Support

For Prefill-Decode disaggregation, merge prefill and decode embeddings:

```python
def _merge_pd_embeddings(
    decode_emb: torch.Tensor, decode_hid: torch.Tensor,
    prefill_mm: dict[str, Any], device: torch.device,
    expected_total: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge prefill prompt embeddings with decode generated embeddings."""
```

The prefill multimodal output is stored in `streaming_context.bridge_states["pd_prefill_multimodal_output_by_req"]`.

---

## Phase 7: Weight Loading

```python
def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
    loaded_weights = set()
    thinker_weights = []
    talker_weights = []
    code2wav_weights = []

    # Separate weights by component prefix
    for k, v in weights:
        if k.startswith("thinker."):
            thinker_weights.append((k, v))
        elif k.startswith("talker."):
            talker_weights.append((k, v))
        elif k.startswith("code2wav."):
            code2wav_weights.append((k, v))

    # Load each component's weights (strip prefix, then re-add to tracking)
    if hasattr(self, 'thinker') and thinker_weights:
        thinker_loaded = self.thinker.load_weights(thinker_weights)
        thinker_loaded = add_prefix_to_loaded_weights(thinker_loaded, "thinker")
        loaded_weights.update(thinker_loaded)

    if hasattr(self, 'talker') and talker_weights:
        talker_loaded = self.talker.load_weights(talker_weights)
        talker_loaded = add_prefix_to_loaded_weights(talker_loaded, "talker")
        loaded_weights.update(talker_loaded)

    # ... code2wav ...
    return loaded_weights
```

---

## Phase 8: Testing

Write an e2e test at `tests/e2e/offline_inference/test_<model>.py`:

```python
@pytest.mark.core_model
def test_<model>_basic_generation():
    """Test basic text generation with thinker stage."""
    ...

@pytest.mark.core_model
def test_<model>_audio_output():
    """Test full pipeline with audio output."""
    ...

@pytest.mark.core_model
def test_<model>_streaming():
    """Test streaming output (if applicable)."""
    ...
```

### What to test

1. **Basic text generation** — thinker-only, verify output is coherent
2. **Audio output** — full pipeline, verify audio duration and sample rate
3. **Streaming** — test with `stream=True`, verify delta/cumulative contract
4. **Empty/short prompts** — edge cases
5. **Concurrent requests** — verify per-request state isolation

### Benchmarking

- Measure RTF (real-time factor) for audio generation
- Compare quality against reference implementation
- Test across different batch sizes and sequence lengths

---

## Phase 9: Online Serving

### 9.1 Register in `serving_speech.py`

If the model produces audio output, add all integration points in a single commit:

**Point 1** — stage constant:
```python
_YOUR_MODEL_STAGES = {"your_stage_key"}
```

**Point 2** — union into `_TTS_MODEL_STAGES`:
```python
_TTS_MODEL_STAGES = (...) | _YOUR_MODEL_STAGES
```

**Point 3** — model type detection in `_detect_tts_model_type()`:
```python
if model_stage in _YOUR_MODEL_STAGES:
    return "your_model"
```

**Point 4** — validation dispatch:
```python
if self._tts_model_type == "your_model":
    return self._validate_your_model_request(request)
```

**Point 5** — validation + parameter-builder methods.

**Always use the `_tts_model_type` string pattern** — do not add new `_is_*` flags.

### 9.2 Voice Cloning (if supported)

```python
def build_voice_clone_prompt(ref_audio_path, text, codec):
    audio_bytes = Path(ref_audio_path).read_bytes()
    codes = codec.encode(audio_bytes)
    token_ids = [code + codec.vocab_offset for code in codes.flatten().tolist()]
    return [
        {"role": "system", "content": f"<|voice|>{''.join(chr(t) for t in token_ids)}"},
        {"role": "user", "content": text},
    ]
```

---

## Phase 10: Adding a Model Recipe

After implementing and testing your model, add a recipe to the [vllm-project/recipes](https://github.com/vllm-project/recipes) repository.

Include:
1. **Model Overview** — capabilities and architecture
2. **Installation** — step-by-step setup
3. **Usage Examples** — CLI commands
4. **Configuration Details** — key parameters

Recipe location: `OrganizationName/ModelName.md` or `ModelName.md`

---

## Phase 11: Pre-commit and DCO

- Install hooks: `pre-commit install`
- Run before push: `pre-commit run --files <changed-files>`
- Sign every commit: `git commit -s`
- Ensure `git config user.email` matches GitHub account email

---

## Integration Checklist

### Phase 1: Setup
- [ ] Model directory created with `__init__.py`
- [ ] Config classes created with `model_type` registration (if needed)
- [ ] Architecture documented (stages, tokens, codebooks, sample rate)
- [ ] Key constants identified (special token IDs, codebook params)

### Phase 2: Stage Components
- [ ] Thinker subclass created with multimodal interfaces
- [ ] Talker subclass created (if separate stage)
- [ ] Code2Wav created (if separate stage)
- [ ] Multimodal processor registered (if multimodal inputs)
- [ ] Hidden state capture implemented in thinker forward

### Phase 3: Unified Model
- [ ] Main model class orchestrates all stages
- [ ] `model_stage` dispatch in `__init__`
- [ ] `OmniOutput` correctly populated
- [ ] `forward()` delegates to the active stage

### Phase 4: Registration
- [ ] All architecture names registered in `_OMNI_MODELS` in `registry.py`
- [ ] `__init__.py` exports main class

### Phase 5: Stage Config YAML
- [ ] Deploy YAML created with all stages
- [ ] Device assignment correct
- [ ] `input_connectors` configured for inter-stage communication
- [ ] `async_chunk` configured (if streaming needed)
- [ ] Platform-specific overrides if needed

### Phase 6: Stage Input Processors
- [ ] Thinker→Talker processor implemented
- [ ] Talker→Code2Wav processor implemented (if applicable)
- [ ] Async chunk variants implemented (if streaming)
- [ ] PD disaggregation merge handled (if applicable)
- [ ] Prompt length computation correct

### Phase 7: Weight Loading
- [ ] `load_weights()` separates weights by prefix
- [ ] Component weights loaded with correct prefix handling

### Phase 8: Testing
- [ ] Basic generation test passes
- [ ] Audio output test passes (if applicable)
- [ ] Streaming test passes (if applicable)
- [ ] Concurrent request test passes (if applicable)

### Phase 9: Online Serving (if applicable)
- [ ] All serving integration points added (single commit)
- [ ] Voice cloning works (if supported)
- [ ] Streaming API endpoint works
- [ ] E2E online serving test written

### Phase 10: Recipe & Docs
- [ ] Model recipe added to vllm-project/recipes
- [ ] Model listed in supported models doc

### Phase 11: Pre-commit
- [ ] `pre-commit` passes
- [ ] All commits signed with DCO

---

## Reference Files

- [Directory Structure](references/directory-structure.md) — detailed layout with file purposes
- [Stage Input Processors](references/stage-input-processors.md) — detailed patterns for stage transitions
- [Online Serving](references/online-serving.md) — serving integration details

Project reference files:
- **Qwen3-Omni**: `vllm_omni/model_executor/models/qwen3_omni/` — canonical three-stage reference
- **Stage config**: `vllm_omni/deploy/qwen3_omni_moe.yaml`
- **Input processors**: `vllm_omni/model_executor/stage_input_processors/qwen3_omni.py`
- **Registry**: `vllm_omni/model_executor/models/registry.py`
- **OmniOutput**: `vllm_omni/model_executor/models/output_templates.py`

For TTS-specific models (two-stage AR+decoder), see the [add-tts-model](../add-tts-model/SKILL.md) skill.
For diffusion models (image/video generation), see the [add-diffusion-model](../add-diffusion-model/SKILL.md) skill.
