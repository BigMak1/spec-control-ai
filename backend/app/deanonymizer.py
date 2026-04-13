"""De-anonymizer: восстановление PII-токенов в тексте отчёта."""

from __future__ import annotations

import re


def deanonymize(report_text: str, pii_map: dict[str, str]) -> str:
    """Заменяет PII-токены ([PERSON_1], [ADDR_1] и т.д.) на исходные значения.

    Args:
        report_text: Текст отчёта с PII-токенами.
        pii_map: Словарь {токен: исходное_значение}, например {"[PERSON_1]": "Иванов И.И."}.

    Returns:
        Текст с восстановленными персональными данными.
    """
    if not pii_map:
        return report_text

    result = report_text
    # Сортировка по длине токена (длинные первыми) для корректной замены
    # вложенных токенов, например [ADDR_1] перед [ADDR_10]
    for token in sorted(pii_map, key=len, reverse=True):
        result = result.replace(token, pii_map[token])

    return result


def find_unreplaced_tokens(text: str) -> list[str]:
    """Находит PII-токены, оставшиеся в тексте после деанонимизации.

    Полезно для валидации: если токены остались — pii_map был неполным.
    """
    return re.findall(r"\[(?:PERSON|ADDR|TEL|EMAIL|INN)_\d+\]", text)
