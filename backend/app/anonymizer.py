"""PII Anonymizer: анонимизация персональных данных перед отправкой в LLM API."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from natasha import Doc, MorphVocab, NewsEmbedding, NewsNERTagger, Segmenter

# Ленивая инициализация тяжёлых моделей Natasha
_segmenter: Segmenter | None = None
_ner_tagger: NewsNERTagger | None = None
_morph_vocab: MorphVocab | None = None


def _get_natasha():
    global _segmenter, _ner_tagger, _morph_vocab
    if _segmenter is None:
        _segmenter = Segmenter()
        _morph_vocab = MorphVocab()
        emb = NewsEmbedding()
        _ner_tagger = NewsNERTagger(emb)
    return _segmenter, _ner_tagger, _morph_vocab


# Regex-паттерны для структурированных PII
_PHONE_RE = re.compile(
    r"(?:\+7|8)[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}"
)
_EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
_INN_RE = re.compile(r"\b\d{10}(?:\d{2})?\b")

# Паттерн для адресов в проектной документации
_ADDR_RE = re.compile(
    r"(?:г\.\s*[А-ЯЁа-яё\-]+(?:[\s,]+(?:ул|пр|пер|б-р|наб|ш|пл)\.\s*[А-ЯЁа-яё\-\s]+(?:,?\s*д\.\s*\d+[А-ЯЁа-яё]*)?))"
)


@dataclass
class _PIISpan:
    """Найденный фрагмент PII в тексте."""

    start: int
    end: int
    category: str  # PERSON, ADDR, TEL, EMAIL, INN
    text: str


@dataclass
class _Counter:
    """Счётчики для нумерации токенов по категориям."""

    counts: dict[str, int] = field(default_factory=dict)

    def next(self, category: str) -> str:
        self.counts[category] = self.counts.get(category, 0) + 1
        return f"[{category}_{self.counts[category]}]"


def anonymize(raw_text: str) -> tuple[str, dict[str, str]]:
    """Анонимизирует PII в тексте.

    Заменяет: ФИО → [PERSON_N], адреса → [ADDR_N], телефоны → [TEL_N],
    email → [EMAIL_N], ИНН → [INN_N].

    Args:
        raw_text: Исходный текст документа.

    Returns:
        Кортеж (anonymized_text, pii_map).
        pii_map: {токен: исходное_значение}, например {"[PERSON_1]": "Иванов П.С."}.
    """
    spans = _find_all_pii(raw_text)

    if not spans:
        return raw_text, {}

    # Сортируем по позиции, чтобы заменять с конца (не ломая индексы)
    spans.sort(key=lambda s: s.start)

    # Дедупликация перекрывающихся спанов: приоритет более длинным
    merged = _merge_overlapping(spans)

    # Формируем замены
    counter = _Counter()
    pii_map: dict[str, str] = {}
    # Словарь для дедупликации одинаковых значений
    value_to_token: dict[str, str] = {}

    # Заменяем с конца текста, чтобы не сбивать индексы
    result = raw_text
    for span in reversed(merged):
        normalized = span.text.strip()
        if normalized in value_to_token:
            token = value_to_token[normalized]
        else:
            token = counter.next(span.category)
            pii_map[token] = normalized
            value_to_token[normalized] = token

        result = result[: span.start] + token + result[span.end :]

    return result, pii_map


def _find_all_pii(text: str) -> list[_PIISpan]:
    """Находит все PII-фрагменты в тексте."""
    spans: list[_PIISpan] = []

    # 1. Regex: телефоны
    for m in _PHONE_RE.finditer(text):
        spans.append(_PIISpan(m.start(), m.end(), "TEL", m.group()))

    # 2. Regex: email
    for m in _EMAIL_RE.finditer(text):
        spans.append(_PIISpan(m.start(), m.end(), "EMAIL", m.group()))

    # 3. Regex: ИНН (10 или 12 цифр)
    for m in _INN_RE.finditer(text):
        spans.append(_PIISpan(m.start(), m.end(), "INN", m.group()))

    # 4. Regex: адреса
    for m in _ADDR_RE.finditer(text):
        spans.append(_PIISpan(m.start(), m.end(), "ADDR", m.group()))

    # 5. Natasha NER: ФИО (PER)
    segmenter, ner_tagger, _ = _get_natasha()
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_ner(ner_tagger)

    for span in doc.spans:
        if span.type == "PER":
            clean = _clean_person_span(span.text)
            if clean:
                spans.append(_PIISpan(span.start, span.stop, "PERSON", clean))

    return spans


# Слова из штампов чертежей и технические термины — не ФИО
_STAMP_WORDS = {
    "разраб", "рук", "н.контроль", "гип", "подп", "дата", "изм", "кол",
    "лист", "листов", "формат", "инв", "зам", "взам", "взаи", "копировал",
    "согласовано", "утвердил", "проверил", "стадия", "заземление",
}

# Паттерн для инициалов: "И.О." или "И.О"
_INITIALS_RE = re.compile(r"^[А-ЯЁ]\.[А-ЯЁ]\.?$")


def _clean_person_span(text: str) -> str | None:
    """Извлекает чистое ФИО из NER-спана, фильтруя мусор штампов."""
    words = text.split()
    # Оставляем только слова, похожие на части имени:
    # - Заглавная буква + кириллица (фамилия/имя/отчество)
    # - Инициалы: "П.С.", "И.Г."
    name_parts: list[str] = []
    for w in words:
        w_clean = w.strip(" ,;:")
        if not w_clean:
            continue
        if w_clean.lower().rstrip(".") in _STAMP_WORDS:
            continue
        if _INITIALS_RE.match(w_clean):
            name_parts.append(w_clean)
        elif (
            len(w_clean) >= 4
            and w_clean[0].isupper()
            and all(c.isalpha() or c == "-" for c in w_clean)
        ):
            name_parts.append(w_clean)

    if not name_parts:
        return None
    # Дедупликация подряд идущих одинаковых слов ("Коваль Коваль" → "Коваль")
    deduped = [name_parts[0]]
    for part in name_parts[1:]:
        if part != deduped[-1]:
            deduped.append(part)

    return " ".join(deduped)


def _merge_overlapping(spans: list[_PIISpan]) -> list[_PIISpan]:
    """Убирает перекрывающиеся спаны, оставляя более длинные."""
    if not spans:
        return []

    # Сортировка по началу, затем по длине (длинные первыми)
    spans.sort(key=lambda s: (s.start, -(s.end - s.start)))

    merged: list[_PIISpan] = [spans[0]]
    for span in spans[1:]:
        if span.start < merged[-1].end:
            # Перекрытие — оставляем более длинный (уже в merged)
            continue
        merged.append(span)

    return merged
