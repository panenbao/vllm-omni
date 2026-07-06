# Stage Input Processors Reference

## Thinker → Talker Data Flow

### Non-Streaming Path (full_payload)

```
Thinker forward() completion
  → OmniOutput(text_hidden_states, multimodal_outputs={"0": emb, "24": hid, ...})
  → orchestrator stores in stage_list[0].engine_outputs

thinker2talker_full_payload(transfer_manager, pooling_output, request)
  → Extracts layer 0 (embedding) and layer 24 (hidden states)
  → Builds OmniPayload with:
    - embed.prefill:      [prompt_len, hidden_dim] tensor
    - embed.tts_bos/eos:  scalar embeddings for special tokens
    - hidden_states.output: [prompt_len, hidden_dim] (projection layer)
    - ids.all:            full sequence (prompt + generated) token IDs
    - ids.prompt:         prompt-only token IDs
    - meta.finished:      boolean tensor
  → Returns dict for connector buffer transfer

thinker2talker_token_only(source_outputs, ...) [orchestrator side]
  → Computes prompt length for KV-cache allocation
  → Returns OmniTokensPrompt with [0]*prompt_len
  → Small metadata (speaker, language) forwarded from user prompt
  → Bulk tensors arrive via connector path (_sync_local_stage_payloads)
```

### Streaming Path (async_chunk)

```
thinker2talker_async_chunk(transfer_manager, pooling_output, request, is_finished)
  → Chunk 0: caches prefill embeddings in transfer_manager, returns None
  → Chunk 1+: returns decode embeddings, flushes cached prefill if present
  → Resumable streaming:
    _construct_thinker2talker_streaming_input_async_chunk()
    → New input segment: caches prefill embeddings
    → Decode flush: merges cached + new embeddings
    → Finished: returns final state with finished=True
```

### Payload Structure

```python
OmniPayloadStruct(
    # Embedding tensors
    embed=EmbeddingsStruct(
        prefill=torch.Tensor,    # [prompt_len, hidden_dim]
        decode=torch.Tensor,     # [1, hidden_dim] (single-step decode)
        tts_bos=torch.Tensor,    # BOS token embedding
        tts_eos=torch.Tensor,    # EOS token embedding
        tts_pad=torch.Tensor,    # PAD token embedding
    ),
    # Hidden states (projection layer)
    hidden_states=HiddenStatesStruct(
        output=torch.Tensor,     # [seq_len, hidden_dim]
    ),
    # Token IDs
    ids=IdsStruct(
        all=list[int],           # Full sequence
        prompt=list[int],        # Prompt only
        output=list[int],        # Generated only
    ),
    # Codec codes (talker→code2wav)
    codes=CodesStruct(
        audio=torch.Tensor,      # [num_codebooks * seq_len]
    ),
    # Metadata
    meta=MetaStruct(
        finished=torch.Tensor,   # boolean scalar
        left_context_size=int,   # async chunk context
    ),
)
```

## Talker → Code2Wav Data Flow

### Non-Streaming

```
talker2code2wav_full_payload(transfer_manager, pooling_output, request)
  → Extracts multimodal_output["codes"]["audio"]
  → Filters rows by output token ID validity
  → Flattens: [num_codebooks, seq_len] → [num_codebooks * seq_len]
  → Returns dict for connector buffer

talker2code2wav(source_outputs, ...) [orchestrator side]
  → Extracts codes.audio from multimodal_output
  → Transpose: [8, seq_len] → [seq_len, 8]
  → Flatten to list[int]
  → Returns OmniTokensPrompt with codes as prompt_token_ids
```

### Streaming (async_chunk)

```
talker2code2wav_async_chunk(transfer_manager, pooling_output, request, is_finished)
  → Appends incremental codes to request buffer
  → Waits for chunk boundary (codec_chunk_frames)
  → Builds chunk with left context overlap
  → Returns OmniPayloadStruct(CodesStruct, MetaStruct(left_context_size, finished))
```

### Codec Chunking

```
configured_initial_chunk_size: 4   # larger first chunk for quality
codec_chunk_frames: 25             # subsequent chunk size
codec_left_context_frames: 25      # overlap for smooth boundaries

Chunk 0: [0:4]  (initial, no overlap)
Chunk 1: [4:54] (25 new + 25 left context from chunk 0)
Chunk N: [N*25-25 : N*25+25]
```

## PD Disaggregation Embedding Merge

In PD mode, prefill and decode run on separate engines. The decode engine receives `bridge_states` containing prefill multimodal output.

```python
_merge_pd_embeddings(decode_emb, decode_hid, prefill_mm, device, expected_total)
  → Extracts prefill emb/hid from prefill_mm["hidden_states"]["layers"]
  → Computes overlap: prefill_len + decode_len - expected_total
  → Concatenates: merged = prefill[:prefill_len] + decode[overlap:]
```

The prefill multimodal output is accessed via:
```python
_get_prefill_multimodal_output(request_id, streaming_context)
  → streaming_context.bridge_states["pd_prefill_multimodal_output_by_req"][request_id]
```

## Prompt Length Computation

The talker stage prompt length matches the user+assistant token positions:

```python
_compute_talker_prompt_ids_length(info, device)
  → Counts tokens between <|im_start|>user and <|im_start|>assistant
  → Adds 9 tokens for assistant prefix
  → Returns total prompt length for KV-cache allocation
  → Talker receives [0]*prompt_len as placeholder prompt
```

## Streaming Context Management

Per-request state for streaming token deltas:

```python
_Thinker2TalkerStreamingState:
  last_prompt_len: int
  last_output_len: int
  merged_sequences: list[int]

_Talker2Codec2WavStreamingState:
  last_seq_len: int

# Accessed via bridge_states:
streaming_context.bridge_states["your_model"][request_id]
```

Token deltas are computed by subtracting the previous state's lengths from the
current, ensuring each chunk transfers only newly generated tokens.
