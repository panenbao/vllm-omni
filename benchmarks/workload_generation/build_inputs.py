#!/usr/bin/env python3
"""Convert a workload CSV (token counts) into a request trace ready for vllm-omni.

The output JSONL has one record per CSV row, schema-compatible with what a
trace-replayer (or a custom ``OmniTraceDataset`` inside vllm-omni) needs.
Each record carries:

    {
      "request_id":   int,
      "timestamp":    float,        # seconds since workload start
      "input_tokens": int,          # for language/reason
      "text_tokens":  int,          # for multimodal (text-only part)
      "output_tokens": int,
      "reason_ratio": float | None,
      "mm_items": [                 # one per multimodal item
        {"modality": "image", "h": 1024, "w": 1024, "t": 1, "target_tokens": 1024},
        {"modality": "video", "t": 60, "h": 1024, "w": 1024,
         "fps": 30.0, "duration_s": 2.0, "target_tokens": 2048},
        {"modality": "audio", "duration_s": 4.0, "num_channels": 1, "target_tokens": 100},
        ...
      ]
    }

For ``mm_items``, ``(h, w, t)`` matches vLLM's ``RandomMultiModalDataset`` bucket
key convention ``(height, width, num_frames)`` (``t=1`` for image, ``t>1`` for
video, audio uses ``duration_s`` instead). ``t`` for video is the **synthetic
generator** frame count at ``fps`` (vLLM-omni hard-codes 30), so the visible
clip duration is ``t / fps`` seconds. ``num_channels`` is the WAV channel count
passed to vLLM-omni's ``generate_synthetic_audio`` (1 = mono, 2 = stereo,
5 = 5.1).

Usage::

    python build_inputs.py \\
        --csv workload/multimodal_120_20_experiment.csv \\
        --model-config configs/qwen3_omni.json \\
        --output workload/multimodal_120_20_experiment.jsonl

The CSV is expected to have these columns (any subset is fine):
  - ``request_id``, ``timestamp`` (always)
  - ``input_tokens`` or ``text_tokens`` (text portion)
  - ``output_tokens``
  - ``image_tokens``, ``audio_tokens``, ``video_tokens`` (JSON arrays of ints)
  - ``audio_channels`` (JSON array of ints; one entry per audio item, optional —
    falls back to ``cfg.audio_default_channels`` when missing)
  - ``reason_ratio`` (float; reason category only)
"""

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Allow `from token_shape import ...` regardless of CWD.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from token_shape import (  # noqa: E402
    ModelTokenConfig,
    audio_tokens_to_duration,
    duration_to_audio_tokens,
    hw_to_image_tokens,
    image_tokens_to_hw,
    thw_to_video_tokens,
    video_tokens_to_thw,
)


# ══════════════════════════════════════════════════════════════════════════════
# CSV helpers
# ══════════════════════════════════════════════════════════════════════════════


def _parse_list(s: str | None) -> list[int]:
    """Parse the string in a CSV cell into a list[int].

    Accepts:
      - JSON array form: ``"[712, 689]"``  (new format, preferred)
      - Empty / missing:  ``""`` or ``"[]"``
      - Legacy numpy form: ``"[np.int64(712), np.int64(689)]"`` (best-effort)
    """
    if s is None or s == "" or s == "[]":
        return []
    try:
        v = json.loads(s)  # tolerate JSON with numpy ints
        return [int(x) for x in v]
    except json.JSONDecodeError:
        # Legacy numpy repr — extract ints with a quick regex
        import re
        nums = re.findall(r"np\.int64\((\d+)\)", s)
        if nums:
            return [int(x) for x in nums]
        res=[int(m) for m in re.findall(r"-?\d+", s)]
        return res


def _parse_int(s: str | None, default: int = 0) -> int:
    if s is None or s == "":
        return default
    return int(float(s))  # tolerate "1234.0"


def _parse_float(s: str | None, default: float = 0.0) -> float:
    if s is None or s == "":
        return default
    return float(s)


# ══════════════════════════════════════════════════════════════════════════════
# Build one request record
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class BuildOptions:
    video_num_frames: int | None = None
    image_aspect: float = 1.0
    video_aspect: float = 1.0


def build_record(
    row: dict[str, str],
    cfg: ModelTokenConfig,
    opts: BuildOptions,
) -> dict[str, Any]:
    """Build a single output JSON record from one CSV row."""
    rec: dict[str, Any] = {
        "request_id": _parse_int(row.get("request_id")),
        "timestamp": _parse_float(row.get("timestamp")),
        "output_tokens": _parse_int(row.get("output_tokens")),
    }

    # Text portion: prefer text_tokens (multimodal) over input_tokens (language).
    if "text_tokens" in row and row["text_tokens"] != "":
        rec["text_tokens"] = _parse_int(row["text_tokens"])
    if "input_tokens" in row and row["input_tokens"] != "":
        rec["input_tokens"] = _parse_int(row["input_tokens"])

    if "reason_ratio" in row and row["reason_ratio"] != "":
        rec["reason_ratio"] = _parse_float(row["reason_ratio"])

    mm_items: list[dict[str, Any]] = []
    for tok in _parse_list(row.get("image_tokens")):
        if tok <= 0:
            continue
        h, w = image_tokens_to_hw(tok, cfg, aspect=opts.image_aspect)
        mm_items.append({
            "modality": "image",
            "h": h, "w": w, "t": 1,
            "target_tokens": tok,
            "actual_tokens": hw_to_image_tokens(h, w, cfg),
        })

    for tok in _parse_list(row.get("video_tokens")):
        if tok <= 0:
            continue
        t, h, w = video_tokens_to_thw(
            tok, cfg, num_frames=opts.video_num_frames, aspect=opts.video_aspect
        )
        mm_items.append({
            "modality": "video",
            "h": h, "w": w, "t": t,
            "fps": cfg.gen_fps,
            "duration_s": round(t / cfg.gen_fps, 3),
            "target_tokens": tok,
            "actual_tokens": thw_to_video_tokens(t, h, w, cfg),
        })

    audio_token_list = _parse_list(row.get("audio_tokens"))
    audio_channel_list = _parse_list(row.get("audio_channels"))
    for idx, tok in enumerate(audio_token_list):
        if tok <= 0:
            continue
        dur = audio_tokens_to_duration(tok, cfg)
        # Use per-item channel from CSV when present; otherwise fall back to model default.
        if idx < len(audio_channel_list) and audio_channel_list[idx] > 0:
            num_channels = int(audio_channel_list[idx])
        else:
            num_channels = int(cfg.audio_default_channels)
        mm_items.append({
            "modality": "audio",
            "duration_s": round(dur, 4),
            "num_channels": num_channels,
            "target_tokens": tok,
            "actual_tokens": duration_to_audio_tokens(dur, cfg),
        })

    if mm_items:
        rec["mm_items"] = mm_items
    return rec


# ══════════════════════════════════════════════════════════════════════════════
# Driver + stats
# ══════════════════════════════════════════════════════════════════════════════


def build_jsonl(
    csv_path: str,
    out_path: str,
    cfg: ModelTokenConfig,
    opts: BuildOptions,
) -> dict[str, Any]:
    """Read CSV and write JSONL. Returns summary stats."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    n_rows = 0
    n_mm = {"image": 0, "video": 0, "audio": 0}
    target_actual_diff = {"image": [], "video": [], "audio": []}

    with open(csv_path) as fin, open(out_path, "w") as fout:
        reader = csv.DictReader(fin)
        for row in reader:
            rec = build_record(row, cfg, opts)
            for item in rec.get("mm_items", []):
                mod = item["modality"]
                n_mm[mod] = n_mm.get(mod, 0) + 1
                tgt = item["target_tokens"]
                act = item["actual_tokens"]
                target_actual_diff[mod].append(act - tgt)
            fout.write(json.dumps(rec) + "\n")
            n_rows += 1

    summary = {
        "rows": n_rows,
        "items": dict(n_mm),
        "mean_token_err": {
            mod: (sum(errs) / len(errs)) if errs else 0.0
            for mod, errs in target_actual_diff.items()
        },
        "max_abs_token_err": {
            mod: (max(abs(e) for e in errs) if errs else 0)
            for mod, errs in target_actual_diff.items()
        },
    }
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build a per-request JSONL trace from a workload CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--csv", required=True, help="Input workload CSV (from generate_experiment.py)")
    p.add_argument("--model-config", default=str(_HERE / "configs" / "qwen3_omni.json"),
                   help="Path to model token config JSON (default: configs/qwen3_omni.json)")
    p.add_argument("--output", "-o", default=None,
                   help="Output JSONL path (default: alongside CSV, replacing .csv with .jsonl)")
    p.add_argument("--video-num-frames", type=int, default=None,
                   help="Force every synthetic video to this many frames at gen_fps. "
                        "Default: pick adaptively per request.")
    p.add_argument("--image-aspect", type=float, default=1.0,
                   help="W/H ratio for images (1.0 = square)")
    p.add_argument("--video-aspect", type=float, default=1.0,
                   help="W/H ratio for videos (1.0 = square)")
    p.add_argument("--verify-tokens", action="store_true",
                   help="Print actual vs. target token error stats after building.")
    return p


def main() -> int:
    args = build_parser().parse_args()
    cfg = ModelTokenConfig.from_json(args.model_config)

    out = args.output or args.csv.rsplit(".", 1)[0] + ".jsonl"

    opts = BuildOptions(
        video_num_frames=args.video_num_frames,
        image_aspect=args.image_aspect,
        video_aspect=args.video_aspect,
    )
    summary = build_jsonl(args.csv, out, cfg, opts)

    print(f"Wrote {summary['rows']} records to {out}")
    print(f"Items per modality: {summary['items']}")
    if args.verify_tokens:
        print("\nToken-count error (actual − target):")
        for mod in ("image", "video", "audio"):
            mean = summary["mean_token_err"][mod]
            mx = summary["max_abs_token_err"][mod]
            print(f"  {mod:5s}  mean = {mean:+.1f}   max|err| = {mx}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
