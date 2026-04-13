"""LLM Wrapper: OpenAI SDK → OpenRouter с retry, cost tracking и LangFuse."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from openai import APIError, APITimeoutError, OpenAI, RateLimitError

from backend.app.config import Settings

logger = logging.getLogger(__name__)

# Приблизительные цены за 1M токенов (OpenRouter pricing для Claude Sonnet)
_PRICING: dict[str, tuple[float, float]] = {
    # model_prefix: (input_per_1M, output_per_1M)
    "anthropic/claude-sonnet": (3.0, 15.0),
    "anthropic/claude-haiku": (0.25, 1.25),
}
_DEFAULT_PRICING = (3.0, 15.0)

# Retry: 3 попытки, exp backoff 2/4/8s
_MAX_RETRIES = 3
_BACKOFF_BASE = 2.0


@dataclass
class UsageStats:
    """Статистика использования LLM за сессию."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    calls: int = 0


@dataclass
class LLMClient:
    """Обёртка над OpenAI SDK для работы с OpenRouter.

    Предоставляет retry с exp backoff, подсчёт токенов и стоимости,
    опциональную интеграцию с LangFuse.
    """

    settings: Settings = field(default_factory=Settings)
    usage: UsageStats = field(default_factory=UsageStats)
    _client: OpenAI | None = field(default=None, repr=False)
    _langfuse: object | None = field(default=None, repr=False)

    def __post_init__(self):
        self._client = OpenAI(
            api_key=self.settings.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        self._init_langfuse()

    def _init_langfuse(self):
        """Инициализирует LangFuse, если ключи заданы."""
        if not self.settings.langfuse_public_key or not self.settings.langfuse_secret_key:
            return
        try:
            from langfuse import Langfuse

            self._langfuse = Langfuse(
                public_key=self.settings.langfuse_public_key,
                secret_key=self.settings.langfuse_secret_key,
                host=self.settings.langfuse_host,
            )
        except Exception:
            logger.warning("LangFuse init failed, tracing disabled", exc_info=True)

    def chat(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        tools: list[dict] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        trace_name: str | None = None,
    ) -> dict:
        """Выполняет chat completion с retry и cost tracking.

        Args:
            messages: Список сообщений в формате OpenAI.
            model: Модель (по умолчанию из настроек).
            tools: Описание инструментов для tool-use.
            temperature: Температура генерации.
            max_tokens: Максимум токенов в ответе.
            trace_name: Имя trace в LangFuse.

        Returns:
            Полный response dict от OpenAI SDK.

        Raises:
            APIError: Если все retry исчерпаны.
        """
        model = model or self.settings.openrouter_model
        kwargs: dict = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools

        response = self._call_with_retry(**kwargs)
        self._track_usage(response, model)
        self._trace_langfuse(messages, response, model, trace_name)

        return response

    def _call_with_retry(self, **kwargs) -> object:
        """Вызов API с exp backoff retry на 429 и 5xx."""
        last_error = None
        for attempt in range(_MAX_RETRIES):
            try:
                return self._client.chat.completions.create(**kwargs)
            except (RateLimitError, APITimeoutError, APIError) as e:
                last_error = e
                # Retry только на 429 и 5xx
                if isinstance(e, APIError) and e.status_code is not None and e.status_code < 500:
                    if not isinstance(e, RateLimitError):
                        raise
                wait = _BACKOFF_BASE ** (attempt + 1)
                logger.warning(
                    "LLM API error (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1,
                    _MAX_RETRIES,
                    wait,
                    e,
                )
                time.sleep(wait)
        raise last_error  # type: ignore[misc]

    def _track_usage(self, response: object, model: str):
        """Обновляет статистику токенов и стоимости."""
        usage = getattr(response, "usage", None)
        if not usage:
            return

        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0

        self.usage.prompt_tokens += prompt_tokens
        self.usage.completion_tokens += completion_tokens
        self.usage.total_tokens += prompt_tokens + completion_tokens
        self.usage.calls += 1

        # Подсчёт стоимости
        input_price, output_price = _DEFAULT_PRICING
        for prefix, pricing in _PRICING.items():
            if model.startswith(prefix):
                input_price, output_price = pricing
                break

        cost = (prompt_tokens / 1_000_000) * input_price + (
            completion_tokens / 1_000_000
        ) * output_price
        self.usage.cost_usd += cost

        logger.info(
            "LLM call: %d prompt + %d completion tokens, cost $%.4f (total $%.4f)",
            prompt_tokens,
            completion_tokens,
            cost,
            self.usage.cost_usd,
        )

    def _trace_langfuse(
        self,
        messages: list[dict],
        response: object,
        model: str,
        trace_name: str | None,
    ):
        """Отправляет trace в LangFuse."""
        if not self._langfuse:
            return
        try:
            trace = self._langfuse.trace(name=trace_name or "llm-call")
            usage = getattr(response, "usage", None)
            trace.generation(
                name="chat",
                model=model,
                input=messages,
                output=self.get_response_text(response),
                usage={
                    "input": getattr(usage, "prompt_tokens", 0) if usage else 0,
                    "output": getattr(usage, "completion_tokens", 0) if usage else 0,
                },
            )
        except Exception:
            logger.warning("LangFuse trace failed", exc_info=True)

    def get_response_text(self, response: object) -> str:
        """Извлекает текст ответа из response."""
        if response.choices:
            return response.choices[0].message.content or ""
        return ""

    def get_tool_calls(self, response: object) -> list:
        """Извлекает tool calls из response."""
        if response.choices:
            return response.choices[0].message.tool_calls or []
        return []
