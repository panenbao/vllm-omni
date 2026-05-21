#!/usr/bin/env python3
"""Generate experiment workloads with customizable request characteristics.

Two ways to drive it:

1. CLI (single-client, single- or multi-window, single-rate-step):

     python generate_experiment.py -c multimodal \
         --text-len 500 --image-count 1 --image-tokens 700 \
         --rate 20 --duration 120

2. JSON config (full power — multi-client, per-window distributions,
   piecewise rate schedule):

     python generate_experiment.py --config configs/scenario.json

CSV output:

  request_id, timestamp, <field_1>, <field_2>, ...

List-valued fields (image_tokens, audio_tokens, video_tokens) are written as
JSON arrays (e.g. ``[712, 689]``) so downstream code can ``json.loads`` them
directly.
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

_DEFAULT_SERVEGEN_ROOT = Path("/home/zhongyu/project/ServeGen")
_SERVEGEN_ROOT = Path(os.environ.get("SERVEGEN_ROOT", _DEFAULT_SERVEGEN_ROOT))
if _SERVEGEN_ROOT.exists():
    sys.path.insert(0, str(_SERVEGEN_ROOT))

from servegen import Category, Client, ClientPool  # noqa: E402
from servegen.construct import generate_workload  # noqa: E402

DISTRIBUTIONS = ("fixed", "normal", "exponential", "pareto", "uniform", "categorical")


# ══════════════════════════════════════════════════════════════════════════════
# Spec dataclasses
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class DistSpec:
    """A 1-D distribution spec used to build a discrete PDF over [0, max]."""

    dist: str = "normal"
    loc: float = 0.0
    scale: float = 1.0
    shape: float = 2.5
    max: int | None = None
    # For "categorical": probs over 0..len(probs)-1
    probs: list[float] | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DistSpec":
        return cls(**d)

    def default_max(self) -> int:
        if self.dist == "categorical":
            return len(self.probs or []) - 1
        if self.dist == "fixed":
            return max(int(self.loc) + 1, 1)
        # leave 3*scale headroom for normal-like dists
        return max(int(self.loc * 3), int(self.loc + self.scale * 3), 1)


@dataclass
class WindowSpec:
    """Trace parameters for a single window (start_ts -> arrival pattern)."""

    rate: float = 5.0
    cv: float = 1.5
    pat: str = "Gamma"
    pat_shape: float = 0.5
    pat_scale: float = 1.0


@dataclass
class ClientSpec:
    """A single client: windows of (trace, per-field distributions)."""

    client_id: int = 0
    # window_start_ts -> WindowSpec
    windows: dict[int, WindowSpec] = field(default_factory=lambda: {0: WindowSpec()})
    # window_start_ts -> {field_name -> DistSpec}
    fields: dict[int, dict[str, DistSpec]] = field(default_factory=dict)
    # window_start_ts -> {field_name -> DistSpec} for fields sampled AFTER ServeGen,
    # at CSV write time (e.g. per-audio-item channel counts). Kept separate because
    # ServeGen's construct only recognises a fixed set of fields.
    post_fields: dict[int, dict[str, DistSpec]] = field(default_factory=dict)


@dataclass
class WorkloadSpec:
    """Top-level spec for one workload generation run."""

    category: str = "language"
    duration: int = 60
    seed: int = 42
    # rate_fn passed to ServeGen: ts -> aggregate rate. Single-step by default.
    rate_schedule: dict[int, float] = field(default_factory=lambda: {0: 10.0})
    clients: list[ClientSpec] = field(default_factory=list)
    output: str | None = None


# ══════════════════════════════════════════════════════════════════════════════
# PDF building
# ══════════════════════════════════════════════════════════════════════════════


def build_pdf(spec: DistSpec) -> list[float]:
    """Build a normalized discrete PDF for integer values in [0, max]."""
    max_val = spec.max if spec.max is not None else spec.default_max()
    max_val = max(int(max_val), 0)

    if spec.dist == "categorical":
        probs = list(spec.probs or [])
        if not probs:
            raise ValueError("categorical dist requires non-empty `probs`")
        pdf = np.zeros(max_val + 1)
        n = min(len(probs), max_val + 1)
        pdf[:n] = probs[:n]
    elif spec.dist == "fixed":
        pdf = np.zeros(max_val + 1)
        pdf[min(int(spec.loc), max_val)] = 1.0
    else:
        x = np.arange(max_val + 1, dtype=float)
        if spec.dist == "normal":
            pdf = stats.norm.pdf(x, loc=spec.loc, scale=max(spec.scale, 1e-9))
        elif spec.dist == "exponential":
            pdf = stats.expon.pdf(x, scale=max(spec.scale, 1e-9))
        elif spec.dist == "pareto":
            pdf = stats.pareto.pdf(x + 1, spec.shape, scale=max(spec.scale, 1e-9))
        elif spec.dist == "uniform":
            low = max(0, int(spec.loc - spec.scale))
            high = min(max_val, int(spec.loc + spec.scale))
            pdf = np.zeros(max_val + 1)
            if high >= low:
                pdf[low : high + 1] = 1.0
        else:
            raise ValueError(f"Unknown dist: {spec.dist}. Choose from {DISTRIBUTIONS}")

    total = pdf.sum()
    if total <= 0:
        raise ValueError(f"PDF for {spec.dist} sums to 0. Check parameters.")
    return (pdf / total).tolist()


# ══════════════════════════════════════════════════════════════════════════════
# Client builder
# ══════════════════════════════════════════════════════════════════════════════


# Per-category field schemas. Each entry: field_name -> default DistSpec generator.
# Used both for CLI defaults and for validation.
_CATEGORY_FIELDS: dict[str, list[str]] = {
    "language": ["input_tokens", "output_tokens"],
    "reason": ["input_tokens", "output_tokens", "reason_ratio"],
    "multimodal": [
        "text_tokens",
        "output_tokens",
        "image_tokens",
        "audio_tokens",
        "video_tokens",
        "image_count",
        "audio_count",
        "video_count",
    ],
}


def build_client(spec: ClientSpec, category: str) -> Client:
    """Convert a ClientSpec to a ServeGen Client."""
    if not spec.windows:
        raise ValueError(f"Client {spec.client_id} has no windows")

    expected = set(_CATEGORY_FIELDS[category])
    trace = {}
    dataset = {}
    for ts, win in spec.windows.items():
        trace[ts] = {
            "rate": win.rate,
            "cv": win.cv,
            "pat": (win.pat, (win.pat_shape, win.pat_scale)),
        }
        win_fields = spec.fields.get(ts) or spec.fields.get(next(iter(spec.fields), 0), {})
        if not win_fields:
            raise ValueError(
                f"Client {spec.client_id} window {ts}: no field distributions provided"
            )
        # Build a PDF for every field defined for this window
        pdfs: dict[str, list[float]] = {}
        for name, dist_spec in win_fields.items():
            if name not in expected:
                raise ValueError(
                    f"Field '{name}' is not valid for category '{category}'. "
                    f"Expected one of: {sorted(expected)}"
                )
            pdfs[name] = build_pdf(dist_spec)
        # Reason needs reason_ratio always; default to uniform [0, 1] if absent
        if category == "reason" and "reason_ratio" not in pdfs:
            pdfs["reason_ratio"] = [1.0 / 100] * 100
        dataset[ts] = pdfs

    return Client(client_id=spec.client_id, trace=trace, dataset=dataset)


# ══════════════════════════════════════════════════════════════════════════════
# CLI → WorkloadSpec
# ══════════════════════════════════════════════════════════════════════════════


def _dist_from_cli(prefix: str, args: argparse.Namespace, loc_default: float = 0) -> DistSpec:
    """Pull --{prefix}-{dist,len,scale,shape,max} from argparse Namespace."""
    return DistSpec(
        dist=getattr(args, f"{prefix}_dist"),
        loc=float(getattr(args, f"{prefix}_len", loc_default)),
        scale=float(getattr(args, f"{prefix}_scale")),
        shape=float(getattr(args, f"{prefix}_shape")),
        max=getattr(args, f"{prefix}_max"),
    )


def _count_dist_from_cli(prefix: str, args: argparse.Namespace) -> DistSpec:
    """Build a count distribution: fixed by default, categorical if --{prefix}-count-probs given."""
    probs = getattr(args, f"{prefix}_count_probs", None)
    if probs:
        return DistSpec(dist="categorical", probs=probs, max=len(probs) - 1)
    return DistSpec(
        dist="fixed",
        loc=float(getattr(args, f"{prefix}_count")),
        max=max(int(getattr(args, f"{prefix}_count")) + 1, 1),
    )


def _channels_dist_from_cli(args: argparse.Namespace) -> DistSpec:
    """Build the per-audio-item channel distribution.

    --audio-channels-probs is a comma-separated list of probabilities over
    channel counts 1..N (index 0 is unused so 1-channel maps to probs[1]).
    Falls back to a fixed value of --audio-channels.
    """
    probs = getattr(args, "audio_channels_probs", None)
    if probs:
        return DistSpec(dist="categorical", probs=probs, max=len(probs) - 1)
    return DistSpec(
        dist="fixed",
        loc=float(args.audio_channels),
        max=max(int(args.audio_channels) + 1, 2),
    )


def workload_from_cli(args: argparse.Namespace) -> WorkloadSpec:
    """Build a WorkloadSpec from argparse Namespace (single client, possibly multi-window)."""
    # Parse --windows and --rate (--rate may be comma-separated to match windows)
    windows = [int(w) for w in str(args.windows).split(",")] if args.windows else [0]
    rate_tokens = str(args.rate).split(",") if isinstance(args.rate, str) else [args.rate]
    rate_vals = [float(r) for r in rate_tokens]
    if len(rate_vals) == 1:
        rate_vals = rate_vals * len(windows)
    if len(rate_vals) != len(windows):
        raise ValueError(
            f"--rate has {len(rate_vals)} values but --windows has {len(windows)} entries"
        )

    win_specs = {
        ts: WindowSpec(
            rate=rate, cv=args.cv, pat=args.pat,
            pat_shape=args.pat_shape, pat_scale=args.pat_scale,
        )
        for ts, rate in zip(windows, rate_vals)
    }

    # Build fields per category
    fields_per_window: dict[str, DistSpec] = {}
    post_fields_per_window: dict[str, DistSpec] = {}
    if args.category == "language":
        fields_per_window["input_tokens"] = _dist_from_cli("input", args, args.input_len)
        fields_per_window["output_tokens"] = _dist_from_cli("output", args, args.output_len)
    elif args.category == "reason":
        fields_per_window["input_tokens"] = _dist_from_cli("input", args, args.input_len)
        fields_per_window["output_tokens"] = _dist_from_cli("output", args, args.output_len)
        # reason_ratio defaults to uniform [0, 1] over 100 bins — handled in build_client
    elif args.category == "multimodal":
        fields_per_window["text_tokens"] = _dist_from_cli("input", args, args.input_len)
        fields_per_window["output_tokens"] = _dist_from_cli("output", args, args.output_len)
        fields_per_window["image_tokens"] = _dist_from_cli("image", args, args.image_len)
        fields_per_window["audio_tokens"] = _dist_from_cli("audio", args, args.audio_len)
        fields_per_window["video_tokens"] = _dist_from_cli("video", args, args.video_len)
        fields_per_window["image_count"] = _count_dist_from_cli("image", args)
        fields_per_window["audio_count"] = _count_dist_from_cli("audio", args)
        fields_per_window["video_count"] = _count_dist_from_cli("video", args)
        # Audio channels — sampled per audio item at CSV write time (not pushed through ServeGen).
        post_fields_per_window["audio_channels"] = _channels_dist_from_cli(args)
    else:
        raise ValueError(f"Unknown category: {args.category}")

    # Same field spec for every window in CLI mode (only the rate changes)
    fields_dict = {ts: dict(fields_per_window) for ts in windows}
    post_fields_dict = (
        {ts: dict(post_fields_per_window) for ts in windows}
        if post_fields_per_window
        else {}
    )

    rate_schedule = dict(zip(windows, rate_vals))
    return WorkloadSpec(
        category=args.category,
        duration=args.duration,
        seed=args.seed,
        rate_schedule=rate_schedule,
        clients=[ClientSpec(
            client_id=0, windows=win_specs,
            fields=fields_dict, post_fields=post_fields_dict,
        )],
        output=args.output,
    )


# ══════════════════════════════════════════════════════════════════════════════
# JSON config → WorkloadSpec
# ══════════════════════════════════════════════════════════════════════════════


def workload_from_json(path: str) -> WorkloadSpec:
    """Load a workload from a JSON config file.

    See ``configs/`` for example formats. Schema:

      {
        "category": "multimodal",
        "duration": 120,
        "seed": 42,
        "rate_schedule": {"0": 10.0, "60": 30.0},
        "output": "workload/scenario_a.csv",
        "clients": [
          {
            "client_id": 0,
            "windows": {
              "0":  {"rate": 10.0, "cv": 1.5, "pat": "Gamma",
                     "pat_shape": 0.5, "pat_scale": 1.0},
              "60": {"rate": 30.0, "cv": 2.0, "pat": "Gamma",
                     "pat_shape": 0.5, "pat_scale": 1.0}
            },
            "fields": {
              "0": {
                "text_tokens":   {"dist": "normal", "loc": 500, "scale": 100, "max": 1500},
                "output_tokens": {"dist": "exponential", "loc": 0, "scale": 100, "max": 1000},
                "image_tokens":  {"dist": "normal", "loc": 700, "scale": 100, "max": 1500},
                "image_count":   {"dist": "categorical", "probs": [0.2, 0.6, 0.2]}
              }
            }
          }
        ]
      }
    """
    with open(path) as f:
        data = json.load(f)

    clients: list[ClientSpec] = []
    for cdata in data.get("clients", []):
        windows = {
            int(ts): WindowSpec(**wd) for ts, wd in cdata.get("windows", {}).items()
        }
        fields = {
            int(ts): {name: DistSpec.from_dict(d) for name, d in fdict.items()}
            for ts, fdict in cdata.get("fields", {}).items()
        }
        post_fields = {
            int(ts): {name: DistSpec.from_dict(d) for name, d in fdict.items()}
            for ts, fdict in cdata.get("post_fields", {}).items()
        }
        clients.append(
            ClientSpec(
                client_id=cdata["client_id"], windows=windows,
                fields=fields, post_fields=post_fields,
            )
        )

    rate_schedule = {
        int(ts): float(r) for ts, r in data.get("rate_schedule", {"0": 10.0}).items()
    }

    return WorkloadSpec(
        category=data["category"],
        duration=int(data["duration"]),
        seed=int(data.get("seed", 42)),
        rate_schedule=rate_schedule,
        clients=clients,
        output=data.get("output"),
    )


# ══════════════════════════════════════════════════════════════════════════════
# CSV writer — keep ints as ints, lists as JSON arrays
# ══════════════════════════════════════════════════════════════════════════════


def _to_python(v: Any) -> Any:
    """Convert numpy scalars/arrays to plain Python types for clean serialization."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.ndarray, list, tuple)):
        return [_to_python(x) for x in v]
    return v


def save_workload_csv(
    requests,
    path: str,
    spec: WorkloadSpec | None = None,
) -> None:
    """Save ServeGen requests to CSV. List-valued fields use JSON encoding.

    If ``spec`` is provided and any client has ``post_fields`` (e.g. ``audio_channels``),
    those values are sampled here per request and added as extra CSV columns.
    """
    import csv

    if not requests:
        raise ValueError("No requests to save.")

    # Resolve the per-window post-field PDFs once. We assume a single-client
    # workload here (which matches our CLI; JSON multi-client is unusual for
    # post_fields). For multi-client we use client 0's post_fields.
    post_field_cdfs: dict[int, dict[str, np.ndarray]] = {}
    if spec is not None and spec.clients:
        client0 = spec.clients[0]
        for ts, fdict in client0.post_fields.items():
            post_field_cdfs[int(ts)] = {
                name: np.cumsum(build_pdf(d)) for name, d in fdict.items()
            }
    sorted_window_starts = sorted(post_field_cdfs.keys()) if post_field_cdfs else []

    def _window_for_ts(ts: float) -> int:
        # Pick the latest window start <= ts; fall back to the earliest.
        last = sorted_window_starts[0]
        for w in sorted_window_starts:
            if w <= ts:
                last = w
            else:
                break
        return last

    rng = np.random.default_rng(spec.seed if spec is not None else 0)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    extra_columns = sorted({n for cdfs in post_field_cdfs.values() for n in cdfs})
    fieldnames = ["request_id", "timestamp", *requests[0].data.keys(), *extra_columns]

    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in requests:
            row: dict[str, Any] = {
                "request_id": int(r.request_id),
                "timestamp": float(r.timestamp),
            }
            for k, v in r.data.items():
                v_py = _to_python(v)
                row[k] = json.dumps(v_py) if isinstance(v_py, list) else v_py

            # Post-fields: sample one value per audio item if audio_channels is configured.
            if extra_columns:
                cdfs = post_field_cdfs.get(_window_for_ts(float(r.timestamp)), {})
                audio_tokens = _to_python(r.data.get("audio_tokens", []))
                n_audio = len(audio_tokens) if isinstance(audio_tokens, list) else 0
                for col in extra_columns:
                    if col == "audio_channels":
                        if n_audio == 0 or "audio_channels" not in cdfs:
                            row[col] = json.dumps([])
                        else:
                            cdf = cdfs["audio_channels"]
                            samples = [int(np.searchsorted(cdf, rng.random())) for _ in range(n_audio)]
                            row[col] = json.dumps(samples)
                    else:
                        # Generic post-field: one scalar per request.
                        cdf = cdfs.get(col)
                        row[col] = (
                            int(np.searchsorted(cdf, rng.random())) if cdf is not None else ""
                        )
            writer.writerow(row)


# ══════════════════════════════════════════════════════════════════════════════
# Stats
# ══════════════════════════════════════════════════════════════════════════════


def print_workload_stats(requests) -> None:
    if not requests:
        print("No requests generated.")
        return

    print(f"\n{'=' * 60}")
    print(f"  Workload Statistics")
    print(f"{'=' * 60}")
    print(f"  Total requests:    {len(requests)}")
    print(f"  Time range:        {requests[0].timestamp:.2f} — {requests[-1].timestamp:.2f} s")
    print(f"  Duration:          {requests[-1].timestamp - requests[0].timestamp:.2f} s")

    for field_name in requests[0].data:
        values: list[float] = []
        for r in requests:
            v = _to_python(r.data[field_name])
            if isinstance(v, list):
                values.extend(v)
            else:
                values.append(v)
        if not values:
            continue
        arr = np.array(values, dtype=float)
        per_req = isinstance(_to_python(requests[0].data[field_name]), list)
        print(f"\n  [{field_name}]" + ("  (flattened across items)" if per_req else ""))
        print(f"    Mean:  {arr.mean():.1f}")
        print(f"    P50:   {np.percentile(arr, 50):.1f}")
        print(f"    P95:   {np.percentile(arr, 95):.1f}")
        print(f"    P99:   {np.percentile(arr, 99):.1f}")
        print(f"    Min:   {arr.min():.1f}")
        print(f"    Max:   {arr.max():.1f}")

    print(f"\n{'=' * 60}")


def print_dry_run(spec: WorkloadSpec) -> None:
    """Print the resolved workload spec without running generation."""
    print(f"\n{'=' * 60}")
    print(f"  Dry-run: workload spec")
    print(f"{'=' * 60}")
    print(f"  Category:       {spec.category}")
    print(f"  Duration:       {spec.duration} s")
    print(f"  Seed:           {spec.seed}")
    print(f"  Rate schedule:  {spec.rate_schedule}")
    print(f"  Output path:    {spec.output or '(auto)'}")
    for c in spec.clients:
        print(f"\n  Client {c.client_id}")
        for ts in sorted(c.windows):
            w = c.windows[ts]
            print(
                f"    @t={ts:>5}s  rate={w.rate} cv={w.cv} "
                f"pat={w.pat}({w.pat_shape},{w.pat_scale})"
            )
            fdict = c.fields.get(ts, {})
            for name, d in fdict.items():
                extra = ""
                if d.dist == "categorical":
                    extra = f" probs={d.probs}"
                elif d.dist == "fixed":
                    extra = f" val={int(d.loc)}"
                else:
                    extra = (
                        f" loc={d.loc} scale={d.scale}"
                        + (f" shape={d.shape}" if d.dist == "pareto" else "")
                        + f" max={d.max if d.max is not None else d.default_max()}"
                    )
                print(f"        {name:14s}: {d.dist}{extra}")
    print(f"{'=' * 60}\n")


# ══════════════════════════════════════════════════════════════════════════════
# argparse
# ══════════════════════════════════════════════════════════════════════════════


def _comma_floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate experiment workloads.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    p.add_argument("--config", default=None,
                   help="Path to a JSON config; overrides all CLI fields except --dry-run/--output")
    p.add_argument("--dry-run", action="store_true",
                   help="Print resolved workload spec and exit (no CSV written)")

    p.add_argument("-c", "--category", default="language",
                   choices=["language", "multimodal", "reason"])
    p.add_argument("-d", "--duration", type=int, default=60)
    p.add_argument("-s", "--seed", type=int, default=42)
    p.add_argument("-o", "--output", default=None,
                   help="Output CSV path (default: workload/{CATEGORY}_{DURATION}_{RATE}_experiment.csv)")

    # Rate / arrival
    p.add_argument("-r", "--rate", default="10.0",
                   help="Request rate(s) in req/s. Single value, or comma-separated to match --windows")
    p.add_argument("--windows", default=None,
                   help="Comma-separated window start timestamps, e.g. '0,60,120'. Default: '0'")
    p.add_argument("--cv", type=float, default=1.5)
    p.add_argument("--pat", default="Gamma", choices=["Gamma", "Weibull"])
    p.add_argument("--pat-shape", type=float, default=0.5)
    p.add_argument("--pat-scale", type=float, default=1.0)

    def add_dist_group(prefix: str, default_len: int, default_scale: float,
                       default_dist: str, help_label: str,
                       len_flags: tuple[str, ...] = ()):
        g = p.add_argument_group(f"{help_label} ({prefix})")
        # Primary len/tokens flag(s). The first one is canonical; later are aliases.
        flags = len_flags or (f"--{prefix}-len",)
        g.add_argument(*flags, type=int, default=default_len, dest=f"{prefix}_len")
        g.add_argument(f"--{prefix}-dist", default=default_dist,
                       choices=list(DISTRIBUTIONS), dest=f"{prefix}_dist")
        g.add_argument(f"--{prefix}-scale", type=float, default=default_scale, dest=f"{prefix}_scale")
        g.add_argument(f"--{prefix}-shape", type=float, default=2.5, dest=f"{prefix}_shape")
        g.add_argument(f"--{prefix}-max", type=int, default=None, dest=f"{prefix}_max")

    # Note: --text-len is a multimodal-friendly alias for --input-len.
    add_dist_group("input", 1000, 200, "normal", "Input/text tokens",
                   len_flags=("--input-len", "--text-len"))
    add_dist_group("output", 200, 100, "exponential", "Output tokens")
    # Image/audio/video lens are commonly called "tokens" by users; accept both.
    add_dist_group("image", 700, 100, "normal", "Image tokens (multimodal)",
                   len_flags=("--image-len", "--image-tokens"))
    add_dist_group("audio", 500, 100, "normal", "Audio tokens (multimodal)",
                   len_flags=("--audio-len", "--audio-tokens"))
    add_dist_group("video", 1000, 200, "normal", "Video tokens (multimodal)",
                   len_flags=("--video-len", "--video-tokens"))

    # Counts
    for mod in ("image", "audio", "video"):
        p.add_argument(f"--{mod}-count", type=int,
                       default=1 if mod == "image" else 0,
                       dest=f"{mod}_count")
        p.add_argument(f"--{mod}-count-probs", type=_comma_floats, default=None,
                       dest=f"{mod}_count_probs",
                       help=f"Comma-separated probs over 0..N for {mod} count (categorical). "
                            f"Overrides --{mod}-count.")

    # Audio channels (per audio item; sampled at CSV write time)
    p.add_argument("--audio-channels", type=int, default=1, dest="audio_channels",
                   help="Number of channels per audio item (1=mono, 2=stereo, 5=5.1). Default: 1")
    p.add_argument("--audio-channels-probs", type=_comma_floats, default=None,
                   dest="audio_channels_probs",
                   help="Comma-separated probs over channel counts 0..N (index 0 unused). "
                        "Example '0,0.7,0.2,0,0,0.1' = 70%% mono, 20%% stereo, 10%% 5.1. "
                        "Overrides --audio-channels.")

    # --text-* aliases handled at parse time
    return p


def _resolve_text_aliases(args: argparse.Namespace) -> None:
    """No-op kept for compatibility; argparse handles --text-len via aliases now."""
    return


def main():
    parser = build_parser()
    args = parser.parse_args()
    _resolve_text_aliases(args)

    if args.config:
        spec = workload_from_json(args.config)
        # CLI overrides
        if args.output:
            spec.output = args.output
    else:
        spec = workload_from_cli(args)

    if args.dry_run:
        print_dry_run(spec)
        return

    # Build pool and generate
    category = Category[spec.category.upper()]
    clients = [build_client(c, spec.category) for c in spec.clients]
    pool = ClientPool.from_clients(category, "experiment", clients)

    print_dry_run(spec)
    print(f"Generating workload ({len(spec.clients)} client(s))...")
    requests = generate_workload(
        pool, spec.rate_schedule, duration=spec.duration, seed=spec.seed
    )

    # Resolve output path
    if spec.output:
        out_path = spec.output
    else:
        avg_rate = sum(spec.rate_schedule.values()) / max(len(spec.rate_schedule), 1)
        out_path = f"workload/{spec.category}_{spec.duration}_{int(avg_rate)}_experiment.csv"

    save_workload_csv(requests, out_path, spec=spec)
    print(f"Saved {len(requests)} requests to {out_path}")

    print_workload_stats(requests)


if __name__ == "__main__":
    main()
