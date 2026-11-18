# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public graph scans and caller-owned cleanup."""

import asyncio
import json
import tempfile
import threading
import zipfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from skillspector import graph
from skillspector.providers import reset_provider, use_provider
from tests.graph_scan import run_graph_scan


def _record_scan_temp_dirs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[Path, list[Path]]:
    """Build a zipped skill and create the scan temp dirs under tmp_path, recording them."""
    archive = tmp_path / "skill.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("SKILL.md", "---\nname: demo\ndescription: d\n---\n# Demo\n")
    created: list[Path] = []
    real_mkdtemp = tempfile.mkdtemp

    def recording_mkdtemp(*args: Any, **kwargs: Any) -> str:
        """Create scan temp dirs under tmp_path and remember them."""
        if kwargs.get("prefix") != "skillspector_":
            return real_mkdtemp(*args, **kwargs)
        path = real_mkdtemp(*args, **{**kwargs, "dir": tmp_path})
        created.append(Path(path))
        return path

    monkeypatch.setattr("skillspector.input_handler.tempfile.mkdtemp", recording_mkdtemp)
    return archive, created


async def test_graph_caller_cleans_up_resolved_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive, created = _record_scan_temp_dirs(monkeypatch, tmp_path)

    result = await run_graph_scan(str(archive))

    assert result["execution_successful"] is True
    assert result["temp_dir_for_cleanup"] == str(created[0])
    assert not created[0].exists()


async def test_graph_caller_fails_removes_its_temp_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A scan failing after input resolution removes the temp dir it made."""
    archive, created = _record_scan_temp_dirs(monkeypatch, tmp_path)

    def fail(*args: Any, **kwargs: Any) -> str:
        """Stand in for a failure after the input is materialized."""
        raise RuntimeError("report failed")

    monkeypatch.setattr("skillspector.nodes.report._format_json", fail)

    with pytest.raises(RuntimeError, match="report failed"):
        await run_graph_scan(str(archive), use_llm=False, output_format="json")

    assert created
    assert not any(path.exists() for path in created)


async def test_graph_caller_is_cancelled_removes_its_temp_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cancelling the graph call after input resolution removes the temp dir."""
    archive, created = _record_scan_temp_dirs(monkeypatch, tmp_path)
    reporting = threading.Event()
    release = threading.Event()

    def block(*args: Any, **kwargs: Any) -> str:
        """Hold the report step so the caller can cancel the running scan."""
        reporting.set()
        release.wait(timeout=10)
        raise RuntimeError("released after cancellation")

    monkeypatch.setattr("skillspector.nodes.report._format_json", block)

    scan = asyncio.create_task(run_graph_scan(str(archive), use_llm=False, output_format="json"))
    try:
        assert await asyncio.to_thread(reporting.wait, 10)
        scan.cancel()
        with pytest.raises(asyncio.CancelledError):
            await scan
    finally:
        release.set()

    assert created
    assert not any(path.exists() for path in created)


def test_static_only_graph_keeps_semantic_nodes_without_constructing_a_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Static graph execution skips semantic model construction but keeps nodes wired."""
    capability_probes = 0

    class _CapabilityProvider:
        DEFAULT_MODEL = "test-model"
        SLOT_DEFAULTS = {"meta_analyzer": "test-model"}

        def get_context_length(self, model: str) -> int | None:
            return 4096 if model == "test-model" else None

        def get_max_output_tokens(self, model: str) -> int | None:
            return 128 if model == "test-model" else None

        def resolve_model(self, slot: str = "default") -> str:
            del slot
            return "test-model"

        def resolve_credentials(self) -> tuple[str, str | None] | None:
            return None

        def create_chat_model(
            self,
            model: str,
            *,
            max_tokens: int,
            timeout: float | None = 120,
        ) -> object:
            nonlocal capability_probes
            del model, max_tokens, timeout
            capability_probes += 1
            return object()

    semantic_model_factory = MagicMock(
        side_effect=AssertionError("static-only semantic nodes must not construct a chat model")
    )
    monkeypatch.setattr("skillspector.llm_analyzer_base.get_chat_model", semantic_model_factory)
    token = use_provider(_CapabilityProvider())
    (tmp_path / "SKILL.md").write_text("# Safe skill\n", encoding="utf-8")
    try:
        result = graph.invoke(
            {"skill_path": str(tmp_path), "use_llm": False, "output_format": "json"}
        )
    finally:
        reset_provider(token)

    assert capability_probes == 0
    semantic_model_factory.assert_not_called()
    assert json.loads(result["report_body"])["metadata"]["llm_available"] is False
    semantic_statuses = {
        status["analyzer_id"]: status["status"]
        for status in result["analyzer_status_events"]
        if status["analyzer_id"].startswith("semantic_")  # type: ignore[index]
    }
    assert semantic_statuses == {
        "semantic_security_discovery": "disabled",
        "semantic_developer_intent": "disabled",
        "semantic_quality_policy": "disabled",
    }


async def test_graph_reports_unavailable_semantic_analysis(
    safe_skill_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    get_chat_model = MagicMock(side_effect=RuntimeError("simulated late-bound provider failure"))
    monkeypatch.setattr("skillspector.llm_analyzer_base.get_chat_model", get_chat_model)

    result = await run_graph_scan(str(safe_skill_dir), use_llm=True)

    get_chat_model.assert_called()
    payload = json.loads(result["report_body"])
    assert payload["metadata"]["llm_requested"] is True
    assert payload["metadata"].get("llm_calls_succeeded", 0) == 0
    assert payload["risk_assessment"]["recommendation"] != "SAFE"
    assert payload["analysis_completeness"]["is_complete"] is False
