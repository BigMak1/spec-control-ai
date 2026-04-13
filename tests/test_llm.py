from unittest.mock import MagicMock, patch

import pytest
from openai import RateLimitError

from backend.app.llm import LLMClient, UsageStats


class TestUsageStats:
    def test_defaults(self):
        stats = UsageStats()
        assert stats.total_tokens == 0
        assert stats.cost_usd == 0.0
        assert stats.calls == 0


class TestLLMClientInit:
    @patch("backend.app.llm.OpenAI")
    def test_creates_openai_client(self, mock_openai):
        client = LLMClient()
        mock_openai.assert_called_once_with(
            api_key=client.settings.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
        )

    @patch("backend.app.llm.OpenAI")
    def test_langfuse_disabled_without_keys(self, mock_openai):
        client = LLMClient()
        assert client._langfuse is None


class TestCostTracking:
    @patch("backend.app.llm.OpenAI")
    def test_tracks_token_usage(self, mock_openai):
        client = LLMClient()

        # Мокаем response
        mock_response = MagicMock()
        mock_response.usage.prompt_tokens = 1000
        mock_response.usage.completion_tokens = 500
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "test"

        client._track_usage(mock_response, "anthropic/claude-sonnet")
        assert client.usage.prompt_tokens == 1000
        assert client.usage.completion_tokens == 500
        assert client.usage.total_tokens == 1500
        assert client.usage.cost_usd > 0

    @patch("backend.app.llm.OpenAI")
    def test_cost_calculation_sonnet(self, mock_openai):
        """Стоимость: 1M input × $3 + 1M output × $15 = $18."""
        client = LLMClient()

        mock_response = MagicMock()
        mock_response.usage.prompt_tokens = 1_000_000
        mock_response.usage.completion_tokens = 1_000_000

        client._track_usage(mock_response, "anthropic/claude-sonnet")
        assert abs(client.usage.cost_usd - 18.0) < 0.01

    @patch("backend.app.llm.OpenAI")
    def test_accumulates_across_calls(self, mock_openai):
        client = LLMClient()

        for _ in range(3):
            mock_response = MagicMock()
            mock_response.usage.prompt_tokens = 100
            mock_response.usage.completion_tokens = 50
            client._track_usage(mock_response, "anthropic/claude-sonnet")

        assert client.usage.calls == 3
        assert client.usage.total_tokens == 450


class TestRetry:
    @patch("backend.app.llm.time.sleep")
    @patch("backend.app.llm.OpenAI")
    def test_retries_on_rate_limit(self, mock_openai, mock_sleep):
        client = LLMClient()
        mock_create = client._client.chat.completions.create

        # Первые 2 вызова — RateLimitError, третий — успех
        rate_limit_error = RateLimitError(
            message="rate limited",
            response=MagicMock(status_code=429, headers={}),
            body=None,
        )
        mock_create.side_effect = [
            rate_limit_error,
            rate_limit_error,
            MagicMock(usage=MagicMock(prompt_tokens=10, completion_tokens=5), choices=[]),
        ]

        client._call_with_retry(
            model="anthropic/claude-sonnet",
            messages=[{"role": "user", "content": "test"}],
        )
        assert mock_create.call_count == 3
        assert mock_sleep.call_count == 2

    @patch("backend.app.llm.time.sleep")
    @patch("backend.app.llm.OpenAI")
    def test_raises_after_max_retries(self, mock_openai, mock_sleep):
        client = LLMClient()
        mock_create = client._client.chat.completions.create

        rate_limit_error = RateLimitError(
            message="rate limited",
            response=MagicMock(status_code=429, headers={}),
            body=None,
        )
        mock_create.side_effect = rate_limit_error

        with pytest.raises(RateLimitError):
            client._call_with_retry(
                model="anthropic/claude-sonnet",
                messages=[{"role": "user", "content": "test"}],
            )
        assert mock_create.call_count == 3


class TestResponseHelpers:
    @patch("backend.app.llm.OpenAI")
    def test_get_response_text(self, mock_openai):
        client = LLMClient()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello world"
        assert client.get_response_text(mock_response) == "Hello world"

    @patch("backend.app.llm.OpenAI")
    def test_get_response_text_empty(self, mock_openai):
        client = LLMClient()
        mock_response = MagicMock()
        mock_response.choices = []
        assert client.get_response_text(mock_response) == ""

    @patch("backend.app.llm.OpenAI")
    def test_get_tool_calls(self, mock_openai):
        client = LLMClient()
        mock_response = MagicMock()
        mock_tc = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.tool_calls = [mock_tc]
        assert client.get_tool_calls(mock_response) == [mock_tc]
