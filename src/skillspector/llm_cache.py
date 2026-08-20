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

"""Opt-in on-disk LLM response cache for reproducible re-scans.

LLM outputs are non-deterministic even with greedy decoding (``temperature=0``),
and provider-side model updates can silently shift answers. When
``SKILLSPECTOR_LLM_CACHE_DIR`` is set, each analyzer batch stores its validated
response on disk keyed by a SHA-256 digest of the full request identity:

- provider name
- model label
- structured-output method
- response-schema fingerprint
- sampling configuration (temperature / seed)
- the exact prompt text

A repeat scan of the same skill with the same model therefore replays the
stored result instead of calling the provider again: the findings and the
derived risk score are byte-for-byte reproducible, and no tokens are spent on
re-analysis. Changing any part of the request identity (a different model,
schema, prompt, or sampling setting) produces a different key and a fresh call.

Only *validated* responses are stored: malformed output that fails Pydantic
validation is never cached, so the existing bounded retry / fail-closed
behaviour is preserved.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)

_CACHE_FORMAT_VERSION = 1
_KEY_SHA_NAMESPACE = b"skillspector-llm-cache-v1"


def resolve_llm_cache_dir() -> str | None:
    """Return the cache directory from ``SKILLSPECTOR_LLM_CACHE_DIR`` or ``None``.

    ``None`` disables the cache entirely (the default, preserving current
    behaviour). The directory is created lazily on the first write.
    """
    raw = os.environ.get("SKILLSPECTOR_LLM_CACHE_DIR", "").strip()
    return raw or None


def resolve_llm_cache_max_age_days() -> float | None:
    """Return the optional cache-entry TTL from ``SKILLSPECTOR_LLM_CACHE_MAX_AGE_DAYS``.

    ``None`` (the default) keeps entries forever. A positive value expires
    entries older than that many days; invalid values fall back to ``None``.
    """
    raw = os.environ.get("SKILLSPECTOR_LLM_CACHE_MAX_AGE_DAYS", "").strip()
    if not raw:
        return None
    try:
        days = float(raw)
    except ValueError:
        logger.warning(
            "Invalid SKILLSPECTOR_LLM_CACHE_MAX_AGE_DAYS=%r (not a float); no expiry",
            raw,
        )
        return None
    if days <= 0:
        return None
    return days


def schema_fingerprint(schema: type[BaseModel] | None) -> str:
    """Return a stable fingerprint of a Pydantic response schema."""
    if schema is None:
        return "none"
    try:
        return json.dumps(schema.model_json_schema(), sort_keys=True, separators=(",", ":"))
    except Exception:  # pragma: no cover - defensive; schemas are local constants
        logger.warning("Could not fingerprint response schema %s", schema.__name__)
        return schema.__name__


def _cache_key(*parts: str) -> str:
    """Return a hex SHA-256 digest of the request identity parts."""
    payload = "\x1f".join(parts).encode("utf-8", errors="surrogatepass")
    return hashlib.sha256(_KEY_SHA_NAMESPACE + payload).hexdigest()


class LLMResponseCache:
    """Opt-in on-disk cache mapping request identity -> validated LLM response.

    Thread-safe enough for concurrent analyzer fan-out: reads are atomic
    ``Path.read_text``, writes go through a temp file + ``os.replace`` so a
    concurrent reader never observes a partially-written entry. Cache misses,
    corrupt entries, and expired entries all behave as a normal (uncached)
    call; failures to write are logged and never raised.
    """

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        method: str,
        schema: type[BaseModel] | None,
        sampling: str,
        votes: int = 1,
        cache_dir: str | None = None,
        max_age_days: float | None = None,
    ) -> None:
        self._cache_dir = cache_dir if cache_dir is not None else resolve_llm_cache_dir()
        self._max_age_days = (
            max_age_days if max_age_days is not None else resolve_llm_cache_max_age_days()
        )
        self._identity_prefix = "\n".join(
            (
                provider,
                model,
                method,
                schema_fingerprint(schema),
                sampling,
                f"votes={votes}",
            )
        )
        self._schema = schema

    @property
    def enabled(self) -> bool:
        return bool(self._cache_dir)

    def _entry_path(self, prompt: str) -> Path:
        assert self._cache_dir is not None  # callers check ``enabled`` first
        digest = _cache_key(self._identity_prefix, prompt)
        return Path(self._cache_dir) / f"{digest}.json"

    def _is_fresh(self, path: Path) -> bool:
        if self._max_age_days is None:
            return True
        try:
            age_seconds = time.time() - path.stat().st_mtime
        except OSError:
            return False
        return age_seconds <= self._max_age_days * 86400

    def load(self, prompt: str) -> BaseModel | str | None:
        """Return the cached response for *prompt*, or ``None`` on any miss."""
        if not self.enabled:
            return None
        path = self._entry_path(prompt)
        if not path.is_file() or not self._is_fresh(path):
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.debug("Ignoring unreadable LLM cache entry %s", path)
            return None
        if payload.get("version") != _CACHE_FORMAT_VERSION:
            return None
        kind = payload.get("kind")
        data = payload.get("data")
        if kind == "raw" and isinstance(data, str):
            return data
        if kind == "structured" and isinstance(data, dict) and self._schema is not None:
            try:
                return self._schema.model_validate(data)
            except Exception:
                logger.debug("Ignoring schema-invalid LLM cache entry %s", path)
                return None
        return None

    def store(self, prompt: str, response: BaseModel | str) -> None:
        """Persist a validated response (Pydantic model or raw string)."""
        if not self.enabled:
            return
        if isinstance(response, BaseModel):
            try:
                data: dict[str, Any] | str = response.model_dump(mode="json")
            except Exception:
                logger.debug("Skipping LLM cache write (response not serializable)")
                return
            kind = "structured"
        elif isinstance(response, str):
            data = response
            kind = "raw"
        else:
            logger.debug("Skipping LLM cache write (unsupported response type)")
            return

        path = self._entry_path(prompt)
        entry = {
            "version": _CACHE_FORMAT_VERSION,
            "kind": kind,
            "data": data,
            "created_at": time.time(),
        }
        assert self._cache_dir is not None
        try:
            Path(self._cache_dir).mkdir(parents=True, exist_ok=True)
        except OSError:
            logger.debug("LLM cache directory unavailable: %s", self._cache_dir, exc_info=True)
            return
        tmp_path = path.with_suffix(f".tmp{os.getpid()}{time.monotonic_ns()}")
        try:
            tmp_path.write_text(
                json.dumps(entry, ensure_ascii=False, separators=(",", ":")),
                encoding="utf-8",
            )
            os.replace(tmp_path, path)
        except OSError:
            logger.debug("LLM cache write failed for %s", path, exc_info=True)
            for candidate in (tmp_path, path):
                try:
                    candidate.unlink(missing_ok=True)
                except OSError:
                    pass
