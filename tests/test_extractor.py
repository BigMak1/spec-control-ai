"""Тесты для Parameter Extractor."""

import json
from unittest.mock import MagicMock

import pytest

from backend.app.config import Settings
from backend.app.extractor import (
    _anonymize_section_text,
    _batch_sections,
    _filter_sections,
    _parse_llm_response,
    extract_parameters,
)
from backend.app.llm import LLMClient
from backend.app.schemas import BudgetExceededError, Section, SessionState


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


class TestParseLlmResponse:
    def test_parses_valid_json_array(self):
        text = json.dumps(
            [
                {
                    "name": "Напряжение",
                    "value": "380",
                    "unit": "В",
                    "context": "электроснабжение здания",
                    "source_page": 5,
                    "source_text": "напряжение 380 В",
                }
            ]
        )
        result = _parse_llm_response(text)
        assert len(result) == 1
        assert result[0].name == "Напряжение"
        assert result[0].value == "380"

    def test_parses_json_with_surrounding_text(self):
        text = (
            "Вот параметры:\n"
            '[{"name":"Тип","value":"ВВГ","unit":null,'
            '"context":"кабель","source_page":1,"source_text":"ВВГ"}]'
            "\nГотово."
        )
        result = _parse_llm_response(text)
        assert len(result) == 1

    def test_returns_empty_on_empty_array(self):
        result = _parse_llm_response("[]")
        assert result == []

    def test_skips_invalid_items(self):
        text = json.dumps(
            [
                {
                    "name": "OK",
                    "value": "1",
                    "unit": "м",
                    "context": "c",
                    "source_page": 1,
                    "source_text": "t",
                },
                {"invalid": "item"},
            ]
        )
        result = _parse_llm_response(text)
        assert len(result) == 1

    def test_returns_empty_on_garbage(self):
        result = _parse_llm_response("это не JSON вообще")
        assert result == []


class TestExtractParameters:
    def _make_session(self, sections, cost_usd=0.0):
        return SessionState(
            session_id="test-session",
            sections=sections,
            pii_map={},
            cost_usd=cost_usd,
        )

    def _make_llm_mock(self, responses: list[str]):
        """Создаёт мок LLMClient, возвращающий заданные ответы последовательно."""
        llm = MagicMock(spec=LLMClient)
        llm.settings = Settings()
        mock_responses = []
        for text in responses:
            resp = MagicMock()
            resp.choices = [MagicMock()]
            resp.choices[0].message.content = text
            resp.usage.prompt_tokens = 100
            resp.usage.completion_tokens = 50
            mock_responses.append(resp)
        llm.chat.side_effect = mock_responses
        llm.get_response_text.side_effect = responses
        llm.usage = MagicMock(cost_usd=0.001, total_tokens=150)
        return llm

    def test_happy_path(self):
        sections = [
            Section(
                name="Электроснабжение",
                text="Напряжение сети 380В, кабель ВВГнг-LS " + "x" * 100,
                page_start=5,
                page_end=7,
            )
        ]
        response_json = json.dumps(
            [
                {
                    "name": "Напряжение сети",
                    "value": "380",
                    "unit": "В",
                    "context": "электроснабжение здания",
                    "source_page": 5,
                    "source_text": "Напряжение сети 380В",
                }
            ]
        )
        llm = self._make_llm_mock([response_json])
        session = self._make_session(sections)

        result = extract_parameters(session, llm)
        assert len(result) == 1
        assert result[0].name == "Напряжение сети"
        llm.chat.assert_called_once()

    def test_empty_sections(self):
        llm = self._make_llm_mock([])
        session = self._make_session([])
        result = extract_parameters(session, llm)
        assert result == []
        llm.chat.assert_not_called()

    def test_budget_exceeded(self):
        sections = [Section(name="S", text="x" * 200, page_start=1, page_end=1)]
        llm = self._make_llm_mock([])
        session = self._make_session(sections, cost_usd=1.5)

        with pytest.raises(BudgetExceededError):
            extract_parameters(session, llm)

    def test_retry_on_invalid_json(self):
        sections = [Section(name="S", text="x" * 200, page_start=1, page_end=1)]
        valid_json = json.dumps(
            [
                {
                    "name": "P",
                    "value": "1",
                    "unit": "м",
                    "context": "c",
                    "source_page": 1,
                    "source_text": "t",
                }
            ]
        )
        llm = self._make_llm_mock(["это не json", valid_json])
        session = self._make_session(sections)

        result = extract_parameters(session, llm)
        assert len(result) == 1
        assert llm.chat.call_count == 2

    def test_anonymizes_sections(self):
        sections = [
            Section(
                name="S",
                text="Иванов П.С. проектировал систему напряжением 380В " + "x" * 80,
                page_start=1,
                page_end=1,
            )
        ]
        response_json = json.dumps(
            [
                {
                    "name": "Напряжение",
                    "value": "380",
                    "unit": "В",
                    "context": "c",
                    "source_page": 1,
                    "source_text": "380В",
                }
            ]
        )
        llm = self._make_llm_mock([response_json])
        session = self._make_session(sections)
        session.pii_map = {"[PERSON_1]": "Иванов П.С."}

        extract_parameters(session, llm)

        # Проверяем, что в LLM отправлен анонимизированный текст
        call_args = llm.chat.call_args
        messages = call_args[0][0] if call_args[0] else call_args[1]["messages"]
        user_msg = next(m["content"] for m in messages if m["role"] == "user")
        assert "Иванов П.С." not in user_msg
        assert "[PERSON_1]" in user_msg

    def test_filters_and_batches(self):
        sections = [
            Section(name="Содержание", text="x" * 200, page_start=1, page_end=1),
            Section(name="S1", text="x" * 200, page_start=2, page_end=2),
            Section(name="S2", text="x" * 200, page_start=3, page_end=3),
            Section(name="S3", text="x" * 200, page_start=4, page_end=4),
        ]
        response_json = json.dumps([])
        llm = self._make_llm_mock([response_json])
        session = self._make_session(sections)

        extract_parameters(session, llm)
        # "Содержание" отфильтрована, 3 мелкие секции в 1 батче = 1 LLM-вызов
        assert llm.chat.call_count == 1
