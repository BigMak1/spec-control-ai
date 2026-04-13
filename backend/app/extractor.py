"""Parameter Extractor: извлечение технических параметров из секций документа через LLM."""

from __future__ import annotations

import json
import logging
import re

from backend.app.config import Settings
from backend.app.llm import LLMClient
from backend.app.prompts.extractor import (
    RETRY_SUFFIX,
    SYSTEM_PROMPT,
    make_user_prompt,
)
from backend.app.schemas import BudgetExceededError, Parameter, Section, SessionState

logger = logging.getLogger(__name__)

# Пре-фильтрация: минимальная длина текста секции
_MIN_SECTION_LENGTH = 100

# Стоп-слова в названиях секций (lowercase)
_STOP_NAMES = {
    "содержание",
    "оглавление",
    "состав проекта",
    "общие указания к чертежам",
    "список литературы",
    "нормативные ссылки",
}

# Батчинг
_SMALL_SECTION_THRESHOLD = 500
_DEFAULT_MAX_BATCH_CHARS = 2000

# Retry
_MAX_JSON_RETRIES = 2


def _anonymize_section_text(text: str, pii_map: dict[str, str]) -> str:
    """Заменяет PII-значения в тексте секции на токены из pii_map."""
    result = text
    for token, original in pii_map.items():
        result = result.replace(original, token)
    return result


def _filter_sections(sections: list[Section]) -> list[Section]:
    """Отфильтровывает секции, не содержащие технических параметров."""
    filtered: list[Section] = []
    for section in sections:
        if len(section.text) < _MIN_SECTION_LENGTH:
            continue
        name_lower = section.name.lower().strip()
        if any(stop in name_lower for stop in _STOP_NAMES):
            continue
        filtered.append(section)
    return filtered


def _batch_sections(
    sections: list[Section], max_chars: int = _DEFAULT_MAX_BATCH_CHARS
) -> list[list[Section]]:
    """Группирует секции в батчи для LLM-вызовов."""
    if not sections:
        return []

    batches: list[list[Section]] = []
    current_batch: list[Section] = []
    current_size = 0

    for section in sections:
        section_len = len(section.text)

        # Большая секция — отдельный батч
        if section_len >= _SMALL_SECTION_THRESHOLD:
            if current_batch:
                batches.append(current_batch)
                current_batch = []
                current_size = 0
            batches.append([section])
            continue

        # Если добавление превысит лимит — закрываем текущий батч
        if current_batch and current_size + section_len > max_chars:
            batches.append(current_batch)
            current_batch = []
            current_size = 0

        current_batch.append(section)
        current_size += section_len

    if current_batch:
        batches.append(current_batch)

    return batches


def _format_batch(sections: list[Section]) -> str:
    """Форматирует батч секций для отправки в LLM."""
    parts: list[str] = []
    for section in sections:
        header = f'--- Секция: "{section.name}" (стр. {section.page_start}-{section.page_end}) ---'
        parts.append(f"{header}\n{section.text}")
    return "\n\n".join(parts)


def _parse_llm_response(text: str) -> list[Parameter]:
    """Парсит JSON-массив параметров из ответа LLM."""
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return []

    try:
        items = json.loads(match.group())
    except json.JSONDecodeError:
        return []

    if not isinstance(items, list):
        return []

    parameters: list[Parameter] = []
    for item in items:
        try:
            parameters.append(Parameter(**item))
        except Exception:
            logger.warning("Skipping invalid parameter item: %s", item)
    return parameters


def extract_parameters(session: SessionState, llm: LLMClient) -> list[Parameter]:
    """Извлекает технические параметры из секций документа через LLM.

    Args:
        session: Состояние сессии с заполненными sections и pii_map.
        llm: LLM-клиент для API-вызовов.

    Returns:
        Список извлечённых параметров.

    Raises:
        BudgetExceededError: При превышении лимита стоимости.
    """
    settings = Settings()

    filtered = _filter_sections(session.sections)
    if not filtered:
        return []

    batches = _batch_sections(filtered)
    all_parameters: list[Parameter] = []

    for batch_idx, batch in enumerate(batches):
        # Budget check
        if session.cost_usd >= settings.circuit_breaker_usd:
            raise BudgetExceededError(
                f"Budget exceeded: ${session.cost_usd:.2f} >= ${settings.circuit_breaker_usd:.2f}"
            )

        # Анонимизация текста секций
        anonymized_batch = [
            Section(
                name=s.name,
                text=_anonymize_section_text(s.text, session.pii_map),
                page_start=s.page_start,
                page_end=s.page_end,
            )
            for s in batch
        ]

        batch_text = _format_batch(anonymized_batch)
        user_prompt = make_user_prompt(batch_text)

        # LLM-вызов с retry
        parameters = _call_llm_with_retry(llm, user_prompt, batch_idx)
        all_parameters.extend(parameters)

        # Обновляем стоимость сессии
        session.cost_usd = llm.usage.cost_usd
        session.token_usage = llm.usage.total_tokens

    return all_parameters


def _call_llm_with_retry(
    llm: LLMClient,
    user_prompt: str,
    batch_idx: int,
) -> list[Parameter]:
    """Вызывает LLM с retry при невалидном JSON."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    for attempt in range(_MAX_JSON_RETRIES + 1):
        response = llm.chat(
            messages,
            temperature=0.0,
            trace_name=f"extract-params-batch-{batch_idx}",
        )
        text = llm.get_response_text(response)
        parameters = _parse_llm_response(text)

        if parameters or text.strip() == "[]" or "[]" in text:
            return parameters

        # Retry с дополнительной инструкцией
        logger.warning(
            "Invalid JSON in batch %d (attempt %d/%d): %s",
            batch_idx,
            attempt + 1,
            _MAX_JSON_RETRIES + 1,
            text[:200],
        )
        if attempt < _MAX_JSON_RETRIES:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT + RETRY_SUFFIX},
                {"role": "user", "content": user_prompt},
            ]

    logger.error("Failed to get valid JSON for batch %d after retries", batch_idx)
    return []
