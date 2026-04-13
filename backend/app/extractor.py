"""Parameter Extractor: извлечение технических параметров из секций документа через LLM."""

from __future__ import annotations

import json  # noqa: F401
import logging
import re  # noqa: F401

from backend.app.config import Settings  # noqa: F401
from backend.app.llm import LLMClient  # noqa: F401
from backend.app.prompts.extractor import (  # noqa: F401
    RETRY_SUFFIX,
    SYSTEM_PROMPT,
    make_user_prompt,
)
from backend.app.schemas import BudgetExceededError, Parameter, Section, SessionState  # noqa: F401

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
