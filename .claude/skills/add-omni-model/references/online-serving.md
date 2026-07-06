# Online Serving for Omni Models

## Serving Speech Integration

For omni models that produce audio output, integrate into `vllm_omni/entrypoints/openai/serving_speech.py`.

### All 5 Integration Points (single commit)

**Point 1 — Stage constant:**
```python
_YOUR_MODEL_STAGES = {"your_stage_key"}
```

**Point 2 — Union into `_TTS_MODEL_STAGES`:**
```python
_TTS_MODEL_STAGES: set[str] = (
    ...
    | _YOUR_MODEL_STAGES
)
```

**Point 3 — Model type detection in `_detect_tts_model_type()`:**
```python
if model_stage in _YOUR_MODEL_STAGES:
    return "your_model"
```

**Point 4 — Validation dispatch in `_validate_tts_request()`:**
```python
if self._tts_model_type == "your_model":
    return self._validate_your_model_request(request)
```

**Point 5 — Validation + parameter-builder methods:**
```python
def _validate_your_model_request(self, request) -> str | None:
    if not request.input or not request.input.strip():
        return "Input text cannot be empty"
    return None

def _build_your_model_params(self, request) -> dict:
    params = {"text": [request.input]}
    if request.voice is not None:
        params["voice"] = [request.voice]
    return params
```

Wire `_build_your_model_params` into `_create_tts_request()` alongside existing model-specific builders.

### Important Rules

- **Always use the `_tts_model_type` string pattern** — do not add new `_is_*` flags (the Fish Speech `_is_fish_speech` boolean is legacy)
- **Only extract fields in `_build_your_model_params` that are actually forwarded** — unused extractions fail `ruff F841`
- **For voice-cloning fields** (`ref_audio` → `prompt_audio_path`, `ref_text` → `prompt_text`), add them to the param builder and verify they reach the model call

### Rebase Conflicts

When rebasing onto `main` after another model was merged, `serving_speech.py` will conflict. Resolution: always keep **both** the upstream model's additions and your own — never discard either side.

## E2E Online Serving Test

Create `tests/e2e/online_serving/test_your_model.py`:

### Pitfalls to Avoid

1. **One `OmniServerParams` set per file.** `omni_server` is module-scoped; a second id in the same file forces mid-module teardown/restart and exposes startup races (`APIConnectionError` on the first request post-restart). Split variants into separate files.

2. **No external URL fetches from the server.** CI and some dev hosts can't reach `raw.githubusercontent.com` over TLS. Inline ref audio as `data:audio/wav;base64,...` — the serving layer accepts both URL and data URL.

3. **Use the harness readiness gate.** The fixture waits for HTTP 200 on `/health`; don't add `time.sleep` in tests. If warmup is incomplete, make `/health` return non-200 until ready.

4. **Mark with `@pytest.mark.core_model` + `hardware_test(res={"cuda": "H100"})`** so the test lands in `test-ready.yml` (triggered by the `ready` label).

## Gradio Demo

Create an interactive demo with:
- Text input for generation
- Voice cloning controls (if supported)
- Streaming audio playback
- Downloadable output

## API Response Formats

Support standard TTS response formats: `wav`, `mp3`, `flac`, `pcm`.

## Online Serving E2E Test Checklist

- [ ] All 5 serving integration points in one commit
- [ ] Validation function rejects empty/bad input
- [ ] Parameter builder handles voice/ref_audio
- [ ] All response formats work
- [ ] Streaming endpoint works (`stream=true`)
- [ ] E2E test passes on CI
- [ ] Gradio demo functional
