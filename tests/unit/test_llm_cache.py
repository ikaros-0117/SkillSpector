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

"""Tests for the opt-in on-disk LLM response cache."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from pydantic import BaseModel, Field

from skillspector.llm_cache import (
    LLMResponseCache,
    resolve_llm_cache_dir,
    resolve_llm_cache_max_age_days,
    schema_fingerprint,
)


class _SampleSchema(BaseModel):
    findings: list[str] = Field(default_factory=list)
    confidence: float = 0.5


def _cache(
    tmp_path: Path,
    *,
    provider: str = "openai",
    model: str = "gpt-5.4",
    method: str = "native",
    schema: type[BaseModel] | None = _SampleSchema,
    sampling: str = "temperature=0.0;seed=none",
    votes: int = 1,
    cache_dir: str | None = None,
    max_age_days: float | None = None,
) -> LLMResponseCache:
    return LLMResponseCache(
        provider=provider,
        model=model,
        method=method,
        schema=schema,
        sampling=sampling,
        votes=votes,
        cache_dir=str(tmp_path) if cache_dir is None else cache_dir,
        max_age_days=max_age_days,
    )


class TestResolveEnv:
    def test_unset_disables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SKILLSPECTOR_LLM_CACHE_DIR", raising=False)
        assert resolve_llm_cache_dir() is None

    def test_set_enables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SKILLSPECTOR_LLM_CACHE_DIR", "/tmp/llm-cache")
        assert resolve_llm_cache_dir() == "/tmp/llm-cache"

    def test_max_age_default_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SKILLSPECTOR_LLM_CACHE_MAX_AGE_DAYS", raising=False)
        assert resolve_llm_cache_max_age_days() is None

    def test_max_age_parsed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SKILLSPECTOR_LLM_CACHE_MAX_AGE_DAYS", "7.5")
        assert resolve_llm_cache_max_age_days() == 7.5

    def test_max_age_invalid_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SKILLSPECTOR_LLM_CACHE_MAX_AGE_DAYS", "abc")
        assert resolve_llm_cache_max_age_days() is None


class TestSchemaFingerprint:
    def test_none_schema(self) -> None:
        assert schema_fingerprint(None) == "none"

    def test_stable_for_same_schema(self) -> None:
        assert schema_fingerprint(_SampleSchema) == schema_fingerprint(_SampleSchema)

    def test_differs_across_schemas(self) -> None:
        class Other(BaseModel):
            count: int = 0

        assert schema_fingerprint(_SampleSchema) != schema_fingerprint(Other)


class TestCacheRoundTrip:
    def test_disabled_without_dir(self, tmp_path: Path) -> None:
        cache = _cache(tmp_path, cache_dir="")
        assert cache.enabled is False
        assert cache.load("prompt") is None
        cache.store("prompt", _SampleSchema(findings=["x"]))  # must not raise
        assert list(tmp_path.iterdir()) == []

    def test_structured_round_trip(self, tmp_path: Path) -> None:
        cache = _cache(tmp_path)
        response = _SampleSchema(findings=["secret", "token"], confidence=0.9)
        cache.store("analyze this", response)

        loaded = cache.load("analyze this")
        assert isinstance(loaded, _SampleSchema)
        assert loaded == response

    def test_raw_round_trip(self, tmp_path: Path) -> None:
        cache = _cache(tmp_path, schema=None)
        cache.store("prompt", '{"findings": []}')
        assert cache.load("prompt") == '{"findings": []}'

    def test_different_prompt_misses(self, tmp_path: Path) -> None:
        cache = _cache(tmp_path)
        cache.store("prompt one", _SampleSchema(findings=["a"]))
        assert cache.load("prompt two") is None

    def test_different_model_misses(self, tmp_path: Path) -> None:
        cache = _cache(tmp_path, model="gpt-5.4")
        cache.store("prompt", _SampleSchema(findings=["a"]))
        other = _cache(tmp_path, model="claude-opus-4-6")
        assert other.load("prompt") is None

    def test_different_sampling_misses(self, tmp_path: Path) -> None:
        cache = _cache(tmp_path, sampling="temperature=0.0;seed=none")
        cache.store("prompt", _SampleSchema(findings=["a"]))
        other = _cache(tmp_path, sampling="temperature=0.7;seed=1")
        assert other.load("prompt") is None

    def test_different_votes_misses(self, tmp_path: Path) -> None:
        cache = _cache(tmp_path, votes=1)
        cache.store("prompt", _SampleSchema(findings=["a"]))
        other = _cache(tmp_path, votes=3)
        assert other.load("prompt") is None

    def test_different_schema_misses(self, tmp_path: Path) -> None:
        cache = _cache(tmp_path, schema=_SampleSchema)
        cache.store("prompt", _SampleSchema(findings=["a"]))

        class Other(BaseModel):
            count: int = 0

        other = _cache(tmp_path, schema=Other)
        assert other.load("prompt") is None

    def test_corrupt_entry_ignored(self, tmp_path: Path) -> None:
        cache = _cache(tmp_path)
        cache.store("prompt", _SampleSchema(findings=["a"]))
        entry = next(tmp_path.iterdir())
        entry.write_text("{not json", encoding="utf-8")
        assert cache.load("prompt") is None

    def test_wrong_version_ignored(self, tmp_path: Path) -> None:
        cache = _cache(tmp_path)
        cache.store("prompt", _SampleSchema(findings=["a"]))
        entry = next(tmp_path.iterdir())
        data = json.loads(entry.read_text(encoding="utf-8"))
        data["version"] = 999
        entry.write_text(json.dumps(data), encoding="utf-8")
        assert cache.load("prompt") is None

    def test_schema_invalid_payload_ignored(self, tmp_path: Path) -> None:
        cache = _cache(tmp_path)
        cache.store("prompt", _SampleSchema(findings=["a"]))
        entry = next(tmp_path.iterdir())
        data = json.loads(entry.read_text(encoding="utf-8"))
        data["data"] = {"findings": "not-a-list"}
        entry.write_text(json.dumps(data), encoding="utf-8")
        assert cache.load("prompt") is None

    def test_expired_entry_ignored(self, tmp_path: Path) -> None:
        cache = _cache(tmp_path, max_age_days=1)
        cache.store("prompt", _SampleSchema(findings=["a"]))
        entry = next(tmp_path.iterdir())
        # Backdate mtime beyond the TTL.
        past = time.time() - 2 * 86400
        import os

        os.utime(entry, (past, past))
        assert cache.load("prompt") is None

    def test_fresh_entry_with_ttl_kept(self, tmp_path: Path) -> None:
        cache = _cache(tmp_path, max_age_days=1)
        cache.store("prompt", _SampleSchema(findings=["a"]))
        assert cache.load("prompt") == _SampleSchema(findings=["a"])

    def test_no_leftover_temp_files(self, tmp_path: Path) -> None:
        cache = _cache(tmp_path)
        cache.store("prompt", _SampleSchema(findings=["a"]))
        files = [p.name for p in tmp_path.iterdir()]
        assert len(files) == 1
        assert not files[0].endswith(".tmp")
