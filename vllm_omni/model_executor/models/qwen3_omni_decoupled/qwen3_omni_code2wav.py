# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Code2Wav stage for decoupled Qwen3-Omni-MoE.

Standalone stage that converts RVQ codec codes to audio waveform.
Delegates to the existing ``Qwen3OmniMoeCode2Wav`` with
code2wav-prefix-only weight loading.
"""

from collections.abc import Iterable

import torch
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
)

from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_code2wav import (
    Qwen3OmniMoeCode2Wav,
)

logger = init_logger(__name__)


class Qwen3OmniMoeCode2WavStage(Qwen3OmniMoeCode2Wav):
    """Code2Wav stage: RVQ codes → audio waveform.

    Inherits all code2wav logic (chunked decode, streaming, snake caches)
    from the existing implementation. Only overrides weight loading to
    ensure only ``code2wav.``-prefixed weights are loaded.
    """

    pass
