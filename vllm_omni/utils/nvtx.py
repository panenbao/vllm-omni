"""NVTX instrumentation helpers for vLLM-Omni profiling.

Set ``VLLM_NVTX_SCOPES_FOR_PROFILING=1`` to emit Omni ranges and marks.
When disabled, or when the optional ``nvtx`` package is unavailable, these
helpers become no-ops.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Callable
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

_RANGE_FUNC: Callable[..., contextlib.AbstractContextManager[Any]] | None = None
_MARK_FUNC: Callable[..., Any] | None = None


def _resolve_funcs() -> None:
    global _RANGE_FUNC, _MARK_FUNC
    if _RANGE_FUNC is not None:
        return

    if bool(int(os.environ.get("VLLM_NVTX_SCOPES_FOR_PROFILING", "0"))):
        try:
            import nvtx
        except ModuleNotFoundError:
            logger.warning(
                "VLLM_NVTX_SCOPES_FOR_PROFILING=1 but the 'nvtx' package is "
                "not installed. NVTX profiling will be disabled for this "
                "session."
            )
            _RANGE_FUNC = lambda *args, **kwargs: contextlib.nullcontext()
            _MARK_FUNC = lambda *args, **kwargs: None
            return

        _RANGE_FUNC = nvtx.annotate
        _MARK_FUNC = nvtx.mark
        logger.info("vLLM-Omni NVTX profiling enabled.")
    else:
        _RANGE_FUNC = lambda *args, **kwargs: contextlib.nullcontext()
        _MARK_FUNC = lambda *args, **kwargs: None


def nvtx_range(name: str, **kwargs: Any) -> contextlib.AbstractContextManager[Any]:
    """Return an NVTX range context manager, or a no-op context manager."""
    _resolve_funcs()
    assert _RANGE_FUNC is not None
    return _RANGE_FUNC(name, **kwargs)


def nvtx_mark(name: str, **kwargs: Any) -> None:
    """Emit an NVTX timestamp marker, or no-op when profiling is disabled."""
    _resolve_funcs()
    assert _MARK_FUNC is not None
    _MARK_FUNC(name, **kwargs)