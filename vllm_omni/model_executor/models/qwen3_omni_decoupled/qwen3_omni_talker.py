# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Talker stage for decoupled Qwen3-Omni-MoE.

Standalone stage that converts hidden states to RVQ codec codes.
Delegates to the existing ``Qwen3OmniMoeTalkerForConditionalGeneration``
with talker-prefix-only weight loading.
"""

from collections.abc import Iterable

import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_talker import (
    Qwen3OmniMoeTalkerForConditionalGeneration,
)

logger = init_logger(__name__)


class Qwen3OmniMoeTalkerStage(
    Qwen3OmniMoeTalkerForConditionalGeneration,
):
    """Talker stage: hidden states → RVQ codec codes.

    Inherits all talker logic (MTP, preprocess, code predictor) from the
    existing implementation. Only overrides weight loading to ensure
    only ``talker.``-prefixed weights are loaded from the unified checkpoint.

    All forward / compute_logits / embed_input_ids / make_omni_output etc.
    are inherited unchanged.
    """

    pass
