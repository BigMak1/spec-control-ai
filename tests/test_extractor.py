"""Тесты для Parameter Extractor."""

from backend.app.extractor import _anonymize_section_text, _batch_sections, _filter_sections
from backend.app.schemas import Section


class TestFilterSections:
    def test_filters_short_sections(self):
        sections = [
            Section(name="Короткая", text="мало текста", page_start=1, page_end=1),
            Section(
                name="Нормальная",
                text="x" * 150,
                page_start=2,
                page_end=3,
            ),
        ]
        result = _filter_sections(sections)
        assert len(result) == 1
        assert result[0].name == "Нормальная"

    def test_filters_toc_sections(self):
        sections = [
            Section(
                name="Содержание документа",
                text="x" * 200,
                page_start=1,
                page_end=1,
            ),
            Section(
                name="1. Общие сведения",
                text="x" * 200,
                page_start=2,
                page_end=3,
            ),
        ]
        result = _filter_sections(sections)
        assert len(result) == 1
        assert result[0].name == "1. Общие сведения"

    def test_keeps_all_relevant_sections(self):
        sections = [
            Section(
                name="2.1 Электроснабжение",
                text="Напряжение 380В, кабель ВВГнг " + "x" * 100,
                page_start=5,
                page_end=7,
            ),
        ]
        result = _filter_sections(sections)
        assert len(result) == 1


class TestBatchSections:
    def test_groups_small_sections(self):
        sections = [
            Section(name=f"Секция {i}", text="x" * 200, page_start=i, page_end=i) for i in range(3)
        ]
        batches = _batch_sections(sections, max_chars=2000)
        assert len(batches) == 1
        assert len(batches[0]) == 3

    def test_large_section_separate_batch(self):
        small = Section(name="Малая", text="x" * 200, page_start=1, page_end=1)
        large = Section(name="Большая", text="x" * 600, page_start=2, page_end=5)
        batches = _batch_sections([small, large], max_chars=2000)
        assert len(batches) == 2

    def test_respects_max_chars(self):
        sections = [
            Section(name=f"S{i}", text="x" * 400, page_start=i, page_end=i) for i in range(6)
        ]
        batches = _batch_sections(sections, max_chars=1000)
        assert all(sum(len(s.text) for s in batch) <= 1000 or len(batch) == 1 for batch in batches)


class TestAnonymizeSectionText:
    def test_replaces_pii_values(self):
        text = "Проектировщик Иванов П.С. разработал систему"
        pii_map = {"[PERSON_1]": "Иванов П.С."}
        result = _anonymize_section_text(text, pii_map)
        assert "[PERSON_1]" in result
        assert "Иванов П.С." not in result

    def test_empty_pii_map(self):
        text = "Напряжение 380В"
        result = _anonymize_section_text(text, {})
        assert result == text

    def test_multiple_replacements(self):
        text = "Иванов П.С. (тел. +7 999 123-45-67) проектировал"
        pii_map = {
            "[PERSON_1]": "Иванов П.С.",
            "[TEL_1]": "+7 999 123-45-67",
        }
        result = _anonymize_section_text(text, pii_map)
        assert "Иванов П.С." not in result
        assert "+7 999 123-45-67" not in result
