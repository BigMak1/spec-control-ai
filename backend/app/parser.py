"""Document Parser: извлечение текста и секций из PDF/DOCX."""

from __future__ import annotations

import re
from pathlib import Path

import docx
import fitz

from backend.app.schemas import Section

# Минимальная длина текста — ниже считаем, что текстовый слой отсутствует
MIN_TEXT_LENGTH = 200


class ParseError(Exception):
    """Ошибка парсинга документа."""


def parse_document(file_path: str | Path) -> tuple[str, list[Section]]:
    """Извлекает текст и секции из PDF или DOCX файла.

    Args:
        file_path: Путь к файлу.

    Returns:
        Кортеж (raw_text, sections).

    Raises:
        ParseError: Если формат не поддерживается или текст не извлечён.
        FileNotFoundError: Если файл не найден.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        raw_text, page_texts = _extract_pdf(path)
    elif suffix in (".docx", ".doc"):
        raw_text, page_texts = _extract_docx(path)
    else:
        raise ParseError(f"Неподдерживаемый формат: {suffix}. Поддерживаются PDF и DOCX.")

    if len(raw_text.strip()) < MIN_TEXT_LENGTH:
        raise ParseError(
            "Документ не содержит текстового слоя. "
            "Загрузите текстовый PDF (не скан без OCR)."
        )

    sections = _split_into_sections(raw_text, page_texts)
    return raw_text, sections


def _extract_pdf(path: Path) -> tuple[str, list[tuple[int, str]]]:
    """Извлекает текст из PDF через PyMuPDF.

    Returns:
        (full_text, [(page_number, page_text), ...])
    """
    doc = fitz.open(str(path))
    try:
        page_texts: list[tuple[int, str]] = []
        parts: list[str] = []
        for i, page in enumerate(doc):
            text = page.get_text()
            page_texts.append((i + 1, text))
            parts.append(text)
        return "\n".join(parts), page_texts
    finally:
        doc.close()


def _extract_docx(path: Path) -> tuple[str, list[tuple[int, str]]]:
    """Извлекает текст из DOCX через python-docx.

    DOCX не имеет нативного понятия «страница», поэтому весь текст
    возвращается как одна «страница».
    """
    document = docx.Document(str(path))
    paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
    full_text = "\n".join(paragraphs)
    return full_text, [(1, full_text)]


# Паттерны заголовков секций в проектной документации
_HEADING_PATTERNS = [
    # Нумерованные заголовки: "1. Общие сведения", "2.1 Электроснабжение"
    re.compile(r"^\s*(\d+(?:\.\d+)*)\s*[.\)]\s*(.+)", re.MULTILINE),
    # Буквенные подпункты: "а) характеристика источников"
    re.compile(r"^\s*([а-яё])\)\s*(.+)", re.MULTILINE),
    # Заголовки с ключевыми словами
    re.compile(
        r"^\s*(Подраздел|Раздел|Глава|ГЛАВА|РАЗДЕЛ)\s+[«\"]?(.+?)[»\"]?\s*$",
        re.MULTILINE,
    ),
]


def _find_page_for_position(
    position: int, raw_text: str, page_texts: list[tuple[int, str]]
) -> int:
    """Определяет номер страницы по позиции символа в полном тексте."""
    offset = 0
    for page_num, page_text in page_texts:
        end = offset + len(page_text) + 1  # +1 для \n
        if position < end:
            return page_num
        offset = end
    return page_texts[-1][0] if page_texts else 1


def _split_into_sections(
    raw_text: str, page_texts: list[tuple[int, str]]
) -> list[Section]:
    """Разбивает текст на секции по заголовкам.

    Если заголовки не найдены — возвращает весь текст как одну секцию.
    """
    # Собираем все совпадения заголовков с позициями
    headings: list[tuple[int, str]] = []  # (position, heading_text)

    for pattern in _HEADING_PATTERNS:
        for match in pattern.finditer(raw_text):
            heading_text = match.group(0).strip()
            # Фильтруем короткие «заголовки» — скорее всего мусор
            if len(heading_text) > 10:
                headings.append((match.start(), heading_text))

    # Сортируем по позиции
    headings.sort(key=lambda h: h[0])

    # Дедупликация: убираем заголовки, которые слишком близко (< 5 символов)
    deduped: list[tuple[int, str]] = []
    for pos, text in headings:
        if not deduped or pos - deduped[-1][0] > 5:
            deduped.append((pos, text))
    headings = deduped

    if not headings:
        # Нет заголовков — одна секция
        first_page = page_texts[0][0] if page_texts else 1
        last_page = page_texts[-1][0] if page_texts else 1
        return [
            Section(
                name="Документ",
                text=raw_text,
                page_start=first_page,
                page_end=last_page,
            )
        ]

    sections: list[Section] = []
    for i, (pos, heading) in enumerate(headings):
        # Текст секции — от текущего заголовка до следующего
        if i + 1 < len(headings):
            next_pos = headings[i + 1][0]
            section_text = raw_text[pos:next_pos].strip()
        else:
            section_text = raw_text[pos:].strip()

        if not section_text:
            continue

        page_start = _find_page_for_position(pos, raw_text, page_texts)
        end_pos = pos + len(section_text)
        page_end = _find_page_for_position(end_pos, raw_text, page_texts)

        sections.append(
            Section(
                name=heading[:150],  # Обрезаем слишком длинные заголовки
                text=section_text,
                page_start=page_start,
                page_end=page_end,
            )
        )

    return sections
