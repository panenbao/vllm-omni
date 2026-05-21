"""Token-count ↔ (H, W, T) / duration inversion for Qwen3-Omni-family models.

The vLLM / vllm-omni ``random-mm`` dataset takes multimodal items shaped as
``(height, width, num_frames)``; our workload CSV gives token *counts* per item.
This module provides the closed-form inversion for the Qwen3-Omni
vision/audio encoder so the bridge script (``build_inputs.py``) can produce
inputs that hit the target token counts within rounding.

Forward formulas (Qwen3-Omni-30B, also valid for Qwen2.5-Omni when its
constants are plugged in via :class:`ModelTokenConfig`):

    image:  tokens = (H / p) * (W / p) / m^2

    video:  Let ``D = T_gen / fps_gen`` be the original video duration.
            The processor resamples to ``fps_enc`` so the encoder sees
            ``T_enc = D * fps_enc`` frames, paired by ``temporal_patch_size``:

                tokens = (T_enc / tp) * (H / p) * (W / p) / m^2
                       = (D * fps_enc / tp) * (H / p) * (W / p) / m^2

    audio:  Qwen2-Audio-style: 1s @ 16kHz, hop=160 → 100 mel frames →
            ((100-1)//2 + 1 - 2)//2 + 1 = 25 tokens. Linearised:

                tokens ≈ duration_s * tokens_per_second   (=25 for Qwen3-Omni)

Inversion notes:
  - ``H`` and ``W`` are rounded to multiples of ``p * m`` so the image
    processor doesn't resize/pad and change the resulting token count.
  - For images we default to a square shape; pass ``aspect`` (= W/H) to bias.
  - For videos, callers either fix ``num_frames`` (recommended) or let us pick
    a small even ``num_frames`` that keeps per-frame token count reasonable.
  - ``num_frames`` is the *generator* frame count at ``fps_gen``; that's what
    vllm-omni's ``RandomMultiModalDataset.generate_synthetic_video`` writes
    into the MP4 (default 30 fps), not the encoder fps.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelTokenConfig:
    """Vision/audio token-count parameters for one model family.

    Defaults are for Qwen3-Omni-30B-A3B-Instruct (see ``configs/qwen3_omni.json``).
    """

    name: str = "qwen3_omni"
    # Vision tower
    patch_size: int = 16
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    # Geometry caps (avoid producing absurdly large images)
    min_hw: int = 32
    max_hw: int = 4096
    # Video sampling
    encoder_fps: float = 2.0      # what the processor resamples video to (Qwen3-Omni: 2)
    gen_fps: float = 30.0         # fps used by vllm's generate_synthetic_video
    # Audio adapter (Qwen2-Audio formula → 25 tok/s @ 16 kHz, hop 160)
    audio_tokens_per_second: float = 25.0
    audio_min_duration_s: float = 0.04
    audio_max_duration_s: float = 600.0
    # Default channel count for synthetic audio when the CSV lacks an
    # ``audio_channels`` column. Qwen3-Omni audio encoder mixes to mono, so the
    # actual token count does not depend on this — but vllm-omni's
    # ``generate_synthetic_audio`` still requires a channel count to write the WAV.
    audio_default_channels: int = 1

    # ── classmethods ──────────────────────────────────────────────────────
    @classmethod
    def from_json(cls, path: str | Path) -> "ModelTokenConfig":
        """Load from JSON. Accepts flat keys or a nested {vision, audio} layout."""
        with open(path) as f:
            data = json.load(f)
        flat: dict = {}
        flat.update(data.get("vision", {}))
        flat.update(data.get("audio", {}))
        for k, v in data.items():
            if k not in ("vision", "audio"):
                flat[k] = v
        return cls(**{k: v for k, v in flat.items() if k in cls.__dataclass_fields__})

    # ── derived quantities ────────────────────────────────────────────────
    @property
    def hw_unit(self) -> int:
        """Smallest H/W increment that keeps token math integer."""
        return self.patch_size * self.spatial_merge_size

    @property
    def video_slices_per_gen_frame(self) -> float:
        """How many temporal slices one synthetic-video frame contributes.

        ``slices_per_gen_frame = fps_enc / (fps_gen * tp)``.
        Used in both forward and inverse video formulas.
        """
        return self.encoder_fps / (self.gen_fps * self.temporal_patch_size)


# ══════════════════════════════════════════════════════════════════════════════
# Forward: shape → tokens (also useful for verification)
# ══════════════════════════════════════════════════════════════════════════════


def hw_to_image_tokens(h: int, w: int, cfg: ModelTokenConfig) -> int:
    p, m = cfg.patch_size, cfg.spatial_merge_size
    return (h // p) * (w // p) // (m * m)


def thw_to_video_tokens(t: int, h: int, w: int, cfg: ModelTokenConfig) -> int:
    """``t`` is the synthetic-video frame count at ``cfg.gen_fps``."""
    p, m = cfg.patch_size, cfg.spatial_merge_size
    slices = max(1, int(math.floor(t * cfg.video_slices_per_gen_frame)))
    return slices * (h // p) * (w // p) // (m * m)


def duration_to_audio_tokens(duration_s: float, cfg: ModelTokenConfig) -> int:
    return max(1, int(round(duration_s * cfg.audio_tokens_per_second)))


# ══════════════════════════════════════════════════════════════════════════════
# Inverse: tokens → shape / duration
# ══════════════════════════════════════════════════════════════════════════════


def _round_to_unit(x: float, unit: int, lo: int, hi: int) -> int:
    """Round ``x`` to the nearest positive multiple of ``unit`` within [lo, hi]."""
    q = max(1, int(round(x / unit)))
    val = q * unit
    if val < lo:
        val = ((lo + unit - 1) // unit) * unit
    if val > hi:
        val = (hi // unit) * unit
    return val


def image_tokens_to_hw(
    tokens: int,
    cfg: ModelTokenConfig,
    aspect: float = 1.0,
) -> tuple[int, int]:
    """Invert ``tokens = (H/p) * (W/p) / m^2`` for image inputs.

    Returns ``(H, W)`` both rounded to multiples of ``cfg.hw_unit`` and clamped
    to ``[cfg.min_hw, cfg.max_hw]``. ``aspect`` = W/H ratio (1.0 = square).

    Because one ``hw_unit × hw_unit`` tile contributes exactly 1 token, the
    natural side count is ``sqrt(tokens)`` in units of ``hw_unit``; we split
    that across H and W using ``aspect``.
    """
    if tokens <= 0:
        return cfg.min_hw, cfg.min_hw
    if aspect <= 0:
        raise ValueError("aspect must be positive")

    unit = cfg.hw_unit
    side_units = math.sqrt(tokens)
    h_units = side_units / math.sqrt(aspect)
    w_units = side_units * math.sqrt(aspect)

    h = _round_to_unit(h_units * unit, unit, cfg.min_hw, cfg.max_hw)
    w = _round_to_unit(w_units * unit, unit, cfg.min_hw, cfg.max_hw)
    return h, w


def video_tokens_to_thw(
    tokens: int,
    cfg: ModelTokenConfig,
    num_frames: int | None = None,
    aspect: float = 1.0,
    per_slice_target: int = 512,
) -> tuple[int, int, int]:
    """Invert the video token formula.

    Parameters
    ----------
    tokens
        Target total video tokens.
    num_frames
        If given, the returned ``T`` will be exactly ``num_frames`` (rounded
        up to satisfy the slice-count quantization). Otherwise we pick the
        smallest ``T`` such that each temporal slice carries about
        ``per_slice_target`` tokens.
    aspect
        W/H ratio (1.0 = square).
    per_slice_target
        Used only when ``num_frames`` is None.
    """
    if tokens <= 0:
        return max(1, int(round(1 / cfg.video_slices_per_gen_frame))), cfg.min_hw, cfg.min_hw

    if num_frames is not None:
        slices = max(1, int(math.floor(num_frames * cfg.video_slices_per_gen_frame)))
        t = num_frames
    else:
        slices = max(1, math.ceil(tokens / per_slice_target))
        # frames per slice = 1 / slices_per_gen_frame
        t = max(
            1,
            int(round(slices / cfg.video_slices_per_gen_frame)),
        )

    tokens_per_slice = max(1, int(round(tokens / slices)))
    h, w = image_tokens_to_hw(tokens_per_slice, cfg, aspect=aspect)
    return t, h, w


def audio_tokens_to_duration(tokens: int, cfg: ModelTokenConfig) -> float:
    """Invert ``tokens ≈ duration_s * tokens_per_second``.

    Returned duration is clamped to ``[audio_min_duration_s, audio_max_duration_s]``.
    """
    if tokens <= 0:
        return cfg.audio_min_duration_s
    duration = tokens / cfg.audio_tokens_per_second
    return max(cfg.audio_min_duration_s, min(cfg.audio_max_duration_s, duration))


# ══════════════════════════════════════════════════════════════════════════════
# Sanity self-check when run as a script
# ══════════════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    cfg = ModelTokenConfig()
    print(f"Config: {cfg.name}  patch={cfg.patch_size}  merge={cfg.spatial_merge_size}  "
          f"hw_unit={cfg.hw_unit}  enc_fps={cfg.encoder_fps}  gen_fps={cfg.gen_fps}  "
          f"audio_tps={cfg.audio_tokens_per_second}")
    print(f"video_slices_per_gen_frame = {cfg.video_slices_per_gen_frame:.6f}  "
          f"(= 1 slice per {1/cfg.video_slices_per_gen_frame:.0f} synthetic frames)")
    print()

    print("Image inversion (target → (H, W) → actual tokens):")
    for target in (64, 256, 700, 1024, 2048):
        h, w = image_tokens_to_hw(target, cfg)
        actual = hw_to_image_tokens(h, w, cfg)
        print(f"  {target:>5} → ({h:>4}, {w:>4}) → {actual:>5}  (err {actual - target:+d})")

    print("\nVideo inversion (target → (T, H, W) → actual tokens):")
    for target, nf in [(1024, None), (2048, 60), (4096, 120)]:
        t, h, w = video_tokens_to_thw(target, cfg, num_frames=nf)
        actual = thw_to_video_tokens(t, h, w, cfg)
        print(f"  {target:>5} (nf={nf}) → (T={t:>3}, H={h:>4}, W={w:>4}) → {actual:>5}  (err {actual - target:+d})")

    print("\nAudio inversion (target tokens → duration_s → actual tokens):")
    for target in (25, 100, 500):
        dur = audio_tokens_to_duration(target, cfg)
        actual = duration_to_audio_tokens(dur, cfg)
        print(f"  {target:>5} → {dur:>6.2f}s → {actual:>5}  (err {actual - target:+d})")
