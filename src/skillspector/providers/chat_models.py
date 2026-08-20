# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared constructors for provider-backed LangChain chat models."""

from __future__ import annotations

import logging
import os
from typing import Any
from urllib.parse import urlparse

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

logger = logging.getLogger(__name__)

# Default sampling temperature for LLM analyzers. Greedy decoding (0.0) makes
# repeated analyses of the same skill far more reproducible than the provider
# defaults (which are typically 0.7-1.0). Models that do not accept a
# temperature (e.g. some reasoning models) silently ignore it.
DEFAULT_LLM_TEMPERATURE = 0.0
# Maximum temperature accepted by OpenAI-compatible endpoints.
MAX_LLM_TEMPERATURE = 2.0


def resolve_reasoning_effort() -> str | None:
    """Resolve the optional provider- and model-dependent reasoning effort."""
    reasoning_effort = os.environ.get("SKILLSPECTOR_REASONING_EFFORT", "").strip()
    return reasoning_effort or None


def resolve_sampling_params() -> tuple[float | None, int | None]:
    """Resolve deterministic sampling params from the environment.

    Returns ``(temperature, seed)``:

    - ``temperature`` defaults to :data:`DEFAULT_LLM_TEMPERATURE` (``0.0``,
      greedy decoding) for reproducible analyses. Set
      ``SKILLSPECTOR_LLM_TEMPERATURE`` to another value in ``[0, 2]`` to
      override, or to an empty string to leave the provider default untouched.
    - ``seed`` is passed only to OpenAI-compatible endpoints that support it
      (OpenAI, Azure OpenAI, Ollama, vLLM, ...). It is ``None`` unless
      ``SKILLSPECTOR_LLM_SEED`` is set; combining ``temperature=0`` with a
      fixed ``seed`` gives the strongest determinism those endpoints offer.
      Providers without ``seed`` support (Anthropic, Bedrock) ignore it.

    Invalid values fall back to the defaults and log a warning rather than
    failing the scan.
    """
    temperature: float | None = DEFAULT_LLM_TEMPERATURE
    raw_temperature = os.environ.get("SKILLSPECTOR_LLM_TEMPERATURE")
    if raw_temperature is not None and raw_temperature.strip() == "":
        temperature = None
    elif raw_temperature is not None:
        try:
            temperature = float(raw_temperature)
            if not 0.0 <= temperature <= MAX_LLM_TEMPERATURE:
                logger.warning(
                    "SKILLSPECTOR_LLM_TEMPERATURE=%r outside [0, %.1f]; using %.1f",
                    raw_temperature,
                    MAX_LLM_TEMPERATURE,
                    DEFAULT_LLM_TEMPERATURE,
                )
                temperature = DEFAULT_LLM_TEMPERATURE
        except ValueError:
            logger.warning(
                "Invalid SKILLSPECTOR_LLM_TEMPERATURE=%r (not a float); using %.1f",
                raw_temperature,
                DEFAULT_LLM_TEMPERATURE,
            )
            temperature = DEFAULT_LLM_TEMPERATURE

    seed: int | None = None
    raw_seed = os.environ.get("SKILLSPECTOR_LLM_SEED", "").strip()
    if raw_seed:
        try:
            seed = int(raw_seed)
        except ValueError:
            logger.warning(
                "Invalid SKILLSPECTOR_LLM_SEED=%r (not an int); ignoring seed",
                raw_seed,
            )

    return temperature, seed


def sampling_signature() -> str:
    """Return a stable string identifying the resolved sampling configuration.

    Used in the LLM response cache key so entries produced under different
    sampling settings never collide.
    """
    temperature, seed = resolve_sampling_params()
    temp_label = "default" if temperature is None else repr(temperature)
    return f"temperature={temp_label};seed={seed if seed is not None else 'none'}"


def validate_base_url(url: str | None) -> None:
    """Warn if *url* is not a well-formed http(s) URL.

    Raises nothing — misconfigured URLs will still fail at the HTTP
    layer, but an early warning helps operators catch typos.
    """
    if url is None:
        return
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        logger.warning(
            "Provider base_url %r has scheme %r — expected http or https. "
            "Requests will likely fail.",
            url,
            parsed.scheme or "(empty)",
        )
    if not parsed.netloc:
        logger.warning(
            "Provider base_url %r has no host component. Requests will likely fail.",
            url,
        )


def create_openai_compatible_chat_model(
    *,
    model: str,
    credentials: tuple[str, str | None] | None,
    max_tokens: int,
    timeout: float | None = 120,
    default_headers: dict[str, str] | None = None,
    temperature: float | None = None,
    seed: int | None = None,
) -> BaseChatModel | None:
    """Create ``ChatOpenAI`` for providers serving OpenAI-compatible endpoints.

    ``temperature`` / ``seed`` default to :func:`resolve_sampling_params` (an
    explicit argument wins; leaving it ``None`` uses the resolved value, which
    is ``0.0`` unless ``SKILLSPECTOR_LLM_TEMPERATURE`` is set to an empty
    string).
    """
    if credentials is None:
        return None

    api_key, base_url = credentials
    validate_base_url(base_url)
    resolved_temperature, resolved_seed = resolve_sampling_params()
    if temperature is None:
        temperature = resolved_temperature
    if seed is None:
        seed = resolved_seed
    kwargs: dict[str, Any] = {
        "model": model,
        "base_url": base_url,
        "api_key": SecretStr(api_key),
        "max_completion_tokens": max_tokens,
        "timeout": timeout,
        "default_headers": default_headers,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    if seed is not None:
        kwargs["seed"] = seed
    reasoning_effort = resolve_reasoning_effort()
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    return ChatOpenAI(**kwargs)
