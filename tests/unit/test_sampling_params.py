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

"""Tests for deterministic LLM sampling configuration and provider propagation."""

from __future__ import annotations

import pytest

from skillspector.providers import anthropic as anthropic_pkg
from skillspector.providers import anthropic_proxy as anthropic_proxy_pkg
from skillspector.providers import azure_openai as azure_pkg
from skillspector.providers import bedrock as bedrock_pkg
from skillspector.providers import chat_models
from skillspector.providers.anthropic import AnthropicProvider
from skillspector.providers.anthropic_proxy import AnthropicProxyProvider
from skillspector.providers.azure_openai import AzureOpenAIProvider
from skillspector.providers.bedrock import BedrockProvider
from skillspector.providers.chat_models import (
    create_openai_compatible_chat_model,
    resolve_sampling_params,
    sampling_signature,
)
from skillspector.providers.openai import OpenAIProvider


class TestResolveSamplingParams:
    def test_unset_defaults_to_greedy_temperature_without_seed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("SKILLSPECTOR_LLM_TEMPERATURE", raising=False)
        monkeypatch.delenv("SKILLSPECTOR_LLM_SEED", raising=False)
        assert resolve_sampling_params() == (0.0, None)

    def test_explicit_temperature_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SKILLSPECTOR_LLM_TEMPERATURE", "0.7")
        monkeypatch.delenv("SKILLSPECTOR_LLM_SEED", raising=False)
        assert resolve_sampling_params() == (0.7, None)

    def test_blank_temperature_keeps_provider_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SKILLSPECTOR_LLM_TEMPERATURE", "   ")
        monkeypatch.delenv("SKILLSPECTOR_LLM_SEED", raising=False)
        assert resolve_sampling_params() == (None, None)

    @pytest.mark.parametrize("bad", ["abc", "1.2.3", "nan", "inf"])
    def test_invalid_temperature_falls_back_to_default(
        self, monkeypatch: pytest.MonkeyPatch, bad: str
    ) -> None:
        monkeypatch.setenv("SKILLSPECTOR_LLM_TEMPERATURE", bad)
        monkeypatch.delenv("SKILLSPECTOR_LLM_SEED", raising=False)
        assert resolve_sampling_params() == (0.0, None)

    @pytest.mark.parametrize("bad", ["-1", "2.5", "100"])
    def test_out_of_range_temperature_clamps_to_default(
        self, monkeypatch: pytest.MonkeyPatch, bad: str
    ) -> None:
        monkeypatch.setenv("SKILLSPECTOR_LLM_TEMPERATURE", bad)
        monkeypatch.delenv("SKILLSPECTOR_LLM_SEED", raising=False)
        assert resolve_sampling_params() == (0.0, None)

    def test_seed_parsed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SKILLSPECTOR_LLM_TEMPERATURE", "0")
        monkeypatch.setenv("SKILLSPECTOR_LLM_SEED", "42")
        assert resolve_sampling_params() == (0.0, 42)

    def test_invalid_seed_ignored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SKILLSPECTOR_LLM_TEMPERATURE", "0")
        monkeypatch.setenv("SKILLSPECTOR_LLM_SEED", "not-an-int")
        assert resolve_sampling_params() == (0.0, None)


class TestSamplingSignature:
    def test_signature_reflects_configuration(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SKILLSPECTOR_LLM_TEMPERATURE", raising=False)
        monkeypatch.delenv("SKILLSPECTOR_LLM_SEED", raising=False)
        assert sampling_signature() == "temperature=0.0;seed=none"

        monkeypatch.setenv("SKILLSPECTOR_LLM_TEMPERATURE", "")
        assert sampling_signature() == "temperature=default;seed=none"

        monkeypatch.setenv("SKILLSPECTOR_LLM_SEED", "7")
        assert sampling_signature() == "temperature=default;seed=7"


class TestOpenAICompatiblePropagation:
    def test_temperature_and_seed_passed_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, object] = {}

        def fake_chat_openai(**kwargs: object) -> dict[str, object]:
            captured.update(kwargs)
            return kwargs

        monkeypatch.setattr(chat_models, "ChatOpenAI", fake_chat_openai)
        monkeypatch.delenv("SKILLSPECTOR_LLM_TEMPERATURE", raising=False)
        monkeypatch.delenv("SKILLSPECTOR_LLM_SEED", raising=False)

        create_openai_compatible_chat_model(
            model="gpt-5.4",
            credentials=("sk-x", "http://localhost:1234/v1"),
            max_tokens=123,
        )

        assert captured["temperature"] == 0.0
        assert "seed" not in captured

    def test_seed_passed_when_configured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, object] = {}

        def fake_chat_openai(**kwargs: object) -> dict[str, object]:
            captured.update(kwargs)
            return kwargs

        monkeypatch.setattr(chat_models, "ChatOpenAI", fake_chat_openai)
        monkeypatch.setenv("SKILLSPECTOR_LLM_SEED", "99")

        create_openai_compatible_chat_model(
            model="gpt-5.4",
            credentials=("sk-x", "http://localhost:1234/v1"),
            max_tokens=123,
        )

        assert captured["seed"] == 99
        assert captured["temperature"] == 0.0

    def test_blank_temperature_omits_temperature(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, object] = {}

        def fake_chat_openai(**kwargs: object) -> dict[str, object]:
            captured.update(kwargs)
            return kwargs

        monkeypatch.setattr(chat_models, "ChatOpenAI", fake_chat_openai)
        monkeypatch.setenv("SKILLSPECTOR_LLM_TEMPERATURE", "  ")

        create_openai_compatible_chat_model(
            model="gpt-5.4",
            credentials=("sk-x", "http://localhost:1234/v1"),
            max_tokens=123,
        )

        assert "temperature" not in captured

    def test_openai_provider_passes_temperature(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, object] = {}

        def fake_chat_openai(**kwargs: object) -> dict[str, object]:
            captured.update(kwargs)
            return kwargs

        monkeypatch.setattr(chat_models, "ChatOpenAI", fake_chat_openai)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
        monkeypatch.delenv("SKILLSPECTOR_LLM_TEMPERATURE", raising=False)
        monkeypatch.delenv("SKILLSPECTOR_LLM_SEED", raising=False)

        OpenAIProvider().create_chat_model("gpt-5.4", max_tokens=123)

        assert captured["temperature"] == 0.0


class TestAnthropicPropagation:
    def test_temperature_passed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, object] = {}

        def fake_chat_anthropic(**kwargs: object) -> dict[str, object]:
            captured.update(kwargs)
            return kwargs

        monkeypatch.setattr(anthropic_pkg.provider, "ChatAnthropic", fake_chat_anthropic)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
        monkeypatch.delenv("SKILLSPECTOR_LLM_TEMPERATURE", raising=False)

        AnthropicProvider().create_chat_model("claude-opus-4-6", max_tokens=123)

        assert captured["temperature"] == 0.0
        assert "seed" not in captured

    def test_blank_temperature_omits_temperature(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, object] = {}

        def fake_chat_anthropic(**kwargs: object) -> dict[str, object]:
            captured.update(kwargs)
            return kwargs

        monkeypatch.setattr(anthropic_pkg.provider, "ChatAnthropic", fake_chat_anthropic)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
        monkeypatch.setenv("SKILLSPECTOR_LLM_TEMPERATURE", "")

        AnthropicProvider().create_chat_model("claude-opus-4-6", max_tokens=123)

        assert "temperature" not in captured


class TestAnthropicProxyPropagation:
    def test_temperature_passed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, object] = {}

        def fake_chat_anthropic_proxy(**kwargs: object) -> dict[str, object]:
            captured.update(kwargs)
            return kwargs

        monkeypatch.setattr(
            anthropic_proxy_pkg.provider, "_ChatAnthropicProxy", fake_chat_anthropic_proxy
        )
        monkeypatch.setenv("ANTHROPIC_PROXY_ENDPOINT_URL", "https://proxy.example/v1")
        monkeypatch.setenv("ANTHROPIC_PROXY_API_KEY", "tok")
        monkeypatch.delenv("SKILLSPECTOR_LLM_TEMPERATURE", raising=False)

        AnthropicProxyProvider().create_chat_model("claude-sonnet-4-6", max_tokens=123)

        assert captured["temperature"] == 0.0
        assert "seed" not in captured


class TestAzurePropagation:
    def test_temperature_and_seed_passed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, object] = {}

        def fake_azure_chat_openai(**kwargs: object) -> dict[str, object]:
            captured.update(kwargs)
            return kwargs

        monkeypatch.setattr(azure_pkg.provider, "AzureChatOpenAI", fake_azure_chat_openai)
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "sk-az")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
        monkeypatch.setenv("SKILLSPECTOR_LLM_TEMPERATURE", "0.2")
        monkeypatch.setenv("SKILLSPECTOR_LLM_SEED", "11")

        AzureOpenAIProvider().create_chat_model("gpt-4o", max_tokens=123)

        assert captured["temperature"] == 0.2
        assert captured["seed"] == 11


class TestBedrockPropagation:
    def test_temperature_passed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, object] = {}

        def fake_bedrock_converse(**kwargs: object) -> dict[str, object]:
            captured.update(kwargs)
            return kwargs

        monkeypatch.setattr(bedrock_pkg.provider, "ChatBedrockConverse", fake_bedrock_converse)
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "ak")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "sk")
        monkeypatch.setenv("AWS_REGION", "us-west-2")
        monkeypatch.delenv("AWS_PROFILE", raising=False)
        monkeypatch.delenv("SKILLSPECTOR_LLM_TEMPERATURE", raising=False)

        # boto3 must resolve credentials for the provider to construct a model.
        BedrockProvider().create_chat_model(BedrockProvider.DEFAULT_MODEL, max_tokens=123)

        assert captured["temperature"] == 0.0
