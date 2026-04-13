# Normative Checker (Agentic RAG) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the Normative Checker agent — the only true agent in the system — that autonomously searches normative documents via FAISS, evaluates relevance, and renders verdicts (PASS/FAIL/MANUAL) for each extracted parameter.

**Architecture:** Single agent loop with tool-use. All parameters are checked in one batch. The agent calls 4 tools (search_norms, get_norm_chunk, compare_values, submit_verdict) and submits verdicts incrementally. Dynamic step limit: `min(3 * len(parameters), 15)`.

**Tech Stack:** Python, OpenAI SDK (tool-use), FAISS, sentence-transformers, Pydantic, pytest

---

## File Structure

| Action | File | Responsibility |
|--------|------|---------------|
| Create | `backend/app/tools.py` | Tool schemas (OpenAI format), tool execution dispatch, compare_values logic |
| Create | `backend/app/prompts/checker.py` | System prompt, user prompt builder |
| Create | `backend/app/checker.py` | Agent loop: check_norms(), submit_verdict handling, anti-hallucination |
| Create | `tests/test_checker.py` | Unit tests with mocked LLM |
| Create | `tests/test_checker_eval.py` | Eval test with real LLM on ground_truth data |

---

### Task 1: Tool Schemas and compare_values

**Files:**
- Create: `backend/app/tools.py`
- Test: `tests/test_checker.py` (first part — compare_values only)

- [ ] **Step 1: Write failing tests for compare_values**

Create `tests/test_checker.py`:

```python
"""Tests for Normative Checker."""

import pytest

from backend.app.tools import compare_values


class TestCompareValues:
    def test_gte_pass(self):
        result = compare_values("4.0", "2.5", "gte")
        assert result["match"] is True
        assert result["actual_parsed"] == 4.0
        assert result["required_parsed"] == 2.5

    def test_gte_fail(self):
        result = compare_values("1.5", "2.5", "gte")
        assert result["match"] is False

    def test_lte_pass(self):
        result = compare_values("10", "15", "lte")
        assert result["match"] is True

    def test_lte_fail(self):
        result = compare_values("20", "15", "lte")
        assert result["match"] is False

    def test_eq_pass(self):
        result = compare_values("380", "380", "eq")
        assert result["match"] is True

    def test_eq_fail(self):
        result = compare_values("220", "380", "eq")
        assert result["match"] is False

    def test_contains_pass(self):
        result = compare_values("ВВГнг(А)-LS", "ВВГнг", "contains")
        assert result["match"] is True

    def test_contains_fail(self):
        result = compare_values("АВВГнг(А)-LSLTx", "ВВГнг", "contains")
        assert result["match"] is False

    def test_unparseable_numbers(self):
        result = compare_values("медные жилы", "2.5", "gte")
        assert result["match"] is None
        assert "cannot parse" in result["explanation"]

    def test_units_in_value(self):
        result = compare_values("2.5 мм²", "2.5", "gte")
        assert result["match"] is True
        assert result["actual_parsed"] == 2.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_checker.py::TestCompareValues -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backend.app.tools'`

- [ ] **Step 3: Implement tools.py with compare_values and TOOL_SCHEMAS**

Create `backend/app/tools.py`:

```python
"""Agent tools for Normative Checker: schemas, execution, compare_values."""

from __future__ import annotations

import json
import re

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_norms",
            "description": "Поиск релевантных пунктов нормативных документов по текстовому запросу. Возвращает список чанков с ID, названием норматива, номером раздела, текстом и оценкой релевантности.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Поисковый запрос на русском языке, описывающий искомое требование норматива",
                    },
                    "top_k": {
                        "type": "integer",
                        "default": 5,
                        "description": "Количество результатов (по умолчанию 5)",
                    },
                    "filter_doc": {
                        "type": "string",
                        "description": "Фильтр по названию нормативного документа (например 'ПУЭ' или 'СП 60'). Опционально.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_norm_chunk",
            "description": "Получить полный текст конкретного пункта норматива по chunk_id, включая соседние пункты для контекста.",
            "parameters": {
                "type": "object",
                "properties": {
                    "chunk_id": {
                        "type": "string",
                        "description": "Идентификатор чанка из результатов search_norms",
                    },
                },
                "required": ["chunk_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_values",
            "description": "Детерминированное сравнение двух значений. Для числовых значений выполняет точное сравнение. Для строковых — проверку вхождения.",
            "parameters": {
                "type": "object",
                "properties": {
                    "actual": {
                        "type": "string",
                        "description": "Фактическое значение из документа",
                    },
                    "required": {
                        "type": "string",
                        "description": "Требуемое значение из норматива",
                    },
                    "comparison_type": {
                        "type": "string",
                        "enum": ["gte", "lte", "eq", "contains"],
                        "description": "Тип сравнения: gte (>=), lte (<=), eq (==), contains (вхождение подстроки)",
                    },
                },
                "required": ["actual", "required", "comparison_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_verdict",
            "description": "Подать вердикт по проверке параметра. Вызывай этот инструмент для каждого проверенного параметра. source_chunk_id должен быть реальным ID из результатов search_norms.",
            "parameters": {
                "type": "object",
                "properties": {
                    "parameter_name": {
                        "type": "string",
                        "description": "Название параметра из списка (точно как в задании)",
                    },
                    "status": {
                        "type": "string",
                        "enum": ["PASS", "FAIL", "MANUAL"],
                        "description": "Вердикт: PASS (соответствует), FAIL (нарушение), MANUAL (требует ручной проверки)",
                    },
                    "norm_reference": {
                        "type": "string",
                        "description": "Ссылка на пункт норматива (например 'ПУЭ 7.1.34')",
                    },
                    "norm_requirement": {
                        "type": "string",
                        "description": "Текст требования из норматива",
                    },
                    "source_chunk_id": {
                        "type": "string",
                        "description": "ID чанка из результатов search_norms, подтверждающий вердикт",
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Уверенность в вердикте от 0.0 до 1.0",
                    },
                    "explanation": {
                        "type": "string",
                        "description": "Краткое обоснование вердикта на русском языке",
                    },
                },
                "required": [
                    "parameter_name",
                    "status",
                    "norm_reference",
                    "norm_requirement",
                    "source_chunk_id",
                    "confidence",
                    "explanation",
                ],
            },
        },
    },
]

_TEXT_PREVIEW_MAX = 300


def _parse_number(s: str) -> float | None:
    """Extract first number from a string like '2.5 мм²' -> 2.5."""
    match = re.search(r"-?\d+(?:\.\d+)?", s)
    if match:
        return float(match.group())
    return None


def compare_values(actual: str, required: str, comparison_type: str) -> dict:
    """Deterministic value comparison.

    Returns dict with keys: match (bool|None), actual_parsed, required_parsed, explanation.
    """
    if comparison_type == "contains":
        match = required in actual
        return {
            "match": match,
            "actual_parsed": actual,
            "required_parsed": required,
            "explanation": f"'{required}' {'найдено' if match else 'не найдено'} в '{actual}'",
        }

    actual_num = _parse_number(actual)
    required_num = _parse_number(required)

    if actual_num is None or required_num is None:
        return {
            "match": None,
            "actual_parsed": actual,
            "required_parsed": required,
            "explanation": "cannot parse numerically, manual comparison needed",
        }

    if comparison_type == "gte":
        match = actual_num >= required_num
    elif comparison_type == "lte":
        match = actual_num <= required_num
    elif comparison_type == "eq":
        match = actual_num == required_num
    else:
        return {
            "match": None,
            "actual_parsed": actual,
            "required_parsed": required,
            "explanation": f"unknown comparison_type: {comparison_type}",
        }

    return {
        "match": match,
        "actual_parsed": actual_num,
        "required_parsed": required_num,
        "explanation": f"{actual_num} {'>=' if comparison_type == 'gte' else '<=' if comparison_type == 'lte' else '=='} {required_num} -> {'True' if match else 'False'}",
    }


def execute_search_norms(retriever, args: dict) -> str:
    """Execute search_norms tool call."""
    query = args["query"]
    top_k = args.get("top_k", 5)
    filter_doc = args.get("filter_doc")

    results = retriever.search_norms(query, top_k=top_k, filter_doc=filter_doc)

    formatted = []
    for r in results:
        meta = r["metadata"]
        text_preview = meta.get("text", "")[:_TEXT_PREVIEW_MAX]
        formatted.append({
            "chunk_id": meta["chunk_id"],
            "norm_doc": meta["norm_doc"],
            "section": meta["section"],
            "title": meta.get("title", ""),
            "score": round(r["score"], 3),
            "text_preview": text_preview,
        })

    return json.dumps(formatted, ensure_ascii=False)


def execute_get_norm_chunk(retriever, args: dict) -> str:
    """Execute get_norm_chunk tool call."""
    chunk_id = args["chunk_id"]
    result = retriever.get_norm_chunk(chunk_id)

    if result is None:
        return json.dumps({"error": "chunk_id not found"}, ensure_ascii=False)

    return json.dumps(result, ensure_ascii=False)


def execute_compare_values(args: dict) -> str:
    """Execute compare_values tool call."""
    result = compare_values(args["actual"], args["required"], args["comparison_type"])
    return json.dumps(result, ensure_ascii=False)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_checker.py::TestCompareValues -v`
Expected: all 10 tests PASS

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check backend/app/tools.py tests/test_checker.py --fix
uv run ruff format backend/app/tools.py tests/test_checker.py
git add backend/app/tools.py tests/test_checker.py
git commit -m "feat(checker): add tool schemas and compare_values with tests"
```

---

### Task 2: Checker Prompt

**Files:**
- Create: `backend/app/prompts/checker.py`

- [ ] **Step 1: Create the checker prompt module**

Create `backend/app/prompts/checker.py`:

```python
"""Prompts for Normative Checker agent."""

from __future__ import annotations

from backend.app.schemas import Parameter

SYSTEM_PROMPT = """\
Ты — инженер-нормоконтролёр. Задача: проверить технические параметры \
из проектной документации на соответствие российским нормативам \
(ПУЭ, СП, ГОСТ).

Доступные инструменты:
- search_norms — поиск по нормативной базе
- get_norm_chunk — получить полный текст пункта норматива с контекстом
- compare_values — детерминированное сравнение значений
- submit_verdict — подать вердикт по параметру

Стратегия работы:
1. Для каждого параметра сформулируй поисковый запрос к нормативной базе
2. Оцени релевантность найденных чанков. Если нерелевантны — переформулируй запрос
3. При необходимости запроси соседние чанки через get_norm_chunk для полного контекста
4. Используй compare_values для числовых сравнений
5. Подай вердикт через submit_verdict для КАЖДОГО параметра

Правила вердиктов:
- PASS — параметр соответствует нормативу
- FAIL — параметр нарушает требование норматива
- MANUAL — не удалось найти релевантный норматив или недостаточно уверенности

Обязательные поля в submit_verdict:
- source_chunk_id — реальный ID чанка из результатов search_norms (не придумывай!)
- confidence — от 0.0 до 1.0, насколько ты уверен в вердикте
- explanation — краткое обоснование на русском языке
- norm_reference — ссылка на пункт норматива (например "ПУЭ 7.1.34")
- norm_requirement — текст требования из норматива

Важно:
- Проверь ВСЕ параметры, не пропускай ни одного
- Используй только chunk_id из результатов search_norms
- Группируй поиск: если несколько параметров относятся к одной теме, \
один поисковый запрос может покрыть несколько параметров
"""


def make_user_prompt(parameters: list[Parameter]) -> str:
    """Build user prompt listing all parameters to check."""
    lines = ["Проверь следующие параметры из проектной документации:\n"]

    for i, p in enumerate(parameters, 1):
        unit_str = f" {p.unit}" if p.unit else ""
        lines.append(
            f'{i}. {p.name} = "{p.value}"{unit_str}\n'
            f"   Контекст: {p.context}\n"
            f"   Источник: стр. {p.source_page}\n"
        )

    return "\n".join(lines)
```

- [ ] **Step 2: Lint and commit**

```bash
uv run ruff check backend/app/prompts/checker.py --fix
uv run ruff format backend/app/prompts/checker.py
git add backend/app/prompts/checker.py
git commit -m "feat(checker): add normative checker agent prompt"
```

---

### Task 3: Agent Loop (checker.py)

**Files:**
- Create: `backend/app/checker.py`

- [ ] **Step 1: Write failing tests for the agent loop**

Append to `tests/test_checker.py`:

```python
import json
from unittest.mock import MagicMock, patch

from backend.app.checker import check_norms, _match_parameter
from backend.app.schemas import CheckResult, Parameter, SessionState


def _make_parameter(name="Напряжение сети", value="380", unit="В"):
    return Parameter(
        name=name,
        value=value,
        unit=unit,
        context="электроснабжение здания",
        source_page=5,
        source_text="Напряжение сети 380В",
    )


def _make_session(parameters=None, cost_usd=0.0):
    return SessionState(
        session_id="test-session",
        parameters=parameters or [],
        cost_usd=cost_usd,
    )


def _make_retriever():
    """Create a mock NormRetriever."""
    retriever = MagicMock()
    retriever.search_norms.return_value = [
        {
            "metadata": {
                "chunk_id": "pue_7_1_34_001",
                "norm_doc": "ПУЭ 7-е изд.",
                "section": "7.1.34",
                "title": "Сечения кабелей",
                "page": 142,
                "text": "Минимальное сечение жил кабелей...",
            },
            "score": 0.85,
        }
    ]
    retriever.get_norm_chunk.return_value = {
        "chunk": {
            "chunk_id": "pue_7_1_34_001",
            "norm_doc": "ПУЭ 7-е изд.",
            "section": "7.1.34",
            "title": "Сечения кабелей",
            "page": 142,
            "text": "Минимальное сечение жил кабелей...",
        },
        "prev": None,
        "next": None,
    }
    return retriever


def _mock_response_no_tools(content="Готово, все параметры проверены."):
    """LLM response with no tool calls (finish_reason=stop)."""
    resp = MagicMock()
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = None
    resp.choices = [MagicMock(message=msg, finish_reason="stop")]
    resp.usage = MagicMock(prompt_tokens=100, completion_tokens=50)
    return resp


def _mock_response_with_tools(tool_calls_data: list[tuple[str, dict]]):
    """LLM response with tool calls.

    tool_calls_data: list of (function_name, arguments_dict)
    """
    resp = MagicMock()
    msg = MagicMock()
    msg.content = None
    tool_calls = []
    for i, (name, args) in enumerate(tool_calls_data):
        tc = MagicMock()
        tc.id = f"call_{i}"
        tc.function.name = name
        tc.function.arguments = json.dumps(args, ensure_ascii=False)
        tool_calls.append(tc)
    msg.tool_calls = tool_calls
    resp.choices = [MagicMock(message=msg, finish_reason="tool_calls")]
    resp.usage = MagicMock(prompt_tokens=200, completion_tokens=100)
    return resp


class TestMatchParameter:
    def test_exact_match(self):
        params = [_make_parameter(name="Напряжение сети")]
        result = _match_parameter("Напряжение сети", params, set())
        assert result is not None
        assert result.name == "Напряжение сети"

    def test_fuzzy_match_substring(self):
        params = [_make_parameter(name="Тип кабеля распределительных сетей детского сада")]
        result = _match_parameter("Тип кабеля распределительных сетей", params, set())
        assert result is not None

    def test_fuzzy_match_reverse(self):
        params = [_make_parameter(name="Сечение PE-проводника")]
        result = _match_parameter("Сечение PE-проводника уравнивания потенциалов", params, set())
        assert result is not None

    def test_no_match(self):
        params = [_make_parameter(name="Напряжение сети")]
        result = _match_parameter("Совершенно другой параметр", params, set())
        assert result is None

    def test_skips_already_used(self):
        params = [_make_parameter(name="Напряжение сети")]
        result = _match_parameter("Напряжение сети", params, {"Напряжение сети"})
        assert result is None


class TestCheckNorms:
    def test_no_parameters(self):
        session = _make_session(parameters=[])
        llm = MagicMock()
        retriever = _make_retriever()
        results = check_norms(session, llm, retriever)
        assert results == []
        llm.chat.assert_not_called()

    def test_agent_stops_immediately(self):
        """Agent returns no tool calls -> all params become MANUAL."""
        param = _make_parameter()
        session = _make_session(parameters=[param])
        llm = MagicMock()
        llm.chat.return_value = _mock_response_no_tools()
        llm.get_tool_calls.return_value = []
        llm.usage = MagicMock(cost_usd=0.01, total_tokens=150)
        retriever = _make_retriever()

        results = check_norms(session, llm, retriever)
        assert len(results) == 1
        assert results[0].status == "MANUAL"
        assert "не вынес вердикт" in results[0].explanation

    def test_submit_verdict_saves_result(self):
        """Agent calls submit_verdict -> CheckResult is saved."""
        param = _make_parameter()
        session = _make_session(parameters=[param])
        retriever = _make_retriever()

        verdict_args = {
            "parameter_name": "Напряжение сети",
            "status": "PASS",
            "norm_reference": "ПУЭ 7.1.34",
            "norm_requirement": "380В",
            "source_chunk_id": "pue_7_1_34_001",
            "confidence": 0.9,
            "explanation": "Соответствует",
        }

        llm = MagicMock()
        llm.chat.side_effect = [
            _mock_response_with_tools([("submit_verdict", verdict_args)]),
            _mock_response_no_tools(),
        ]
        llm.get_tool_calls.side_effect = [
            _mock_response_with_tools([("submit_verdict", verdict_args)]).choices[0].message.tool_calls,
            [],
        ]
        llm.usage = MagicMock(cost_usd=0.01, total_tokens=300)

        results = check_norms(session, llm, retriever)
        assert len(results) == 1
        assert results[0].status == "PASS"
        assert results[0].source_chunk_id == "pue_7_1_34_001"

    def test_invalid_chunk_id_rejected(self):
        """submit_verdict with non-existent chunk_id is rejected."""
        param = _make_parameter()
        session = _make_session(parameters=[param])
        retriever = MagicMock()
        retriever.get_norm_chunk.return_value = None  # chunk not found

        verdict_args = {
            "parameter_name": "Напряжение сети",
            "status": "PASS",
            "norm_reference": "ПУЭ 7.1.34",
            "norm_requirement": "380В",
            "source_chunk_id": "fake_chunk_999",
            "confidence": 0.9,
            "explanation": "Соответствует",
        }

        llm = MagicMock()
        llm.chat.side_effect = [
            _mock_response_with_tools([("submit_verdict", verdict_args)]),
            _mock_response_no_tools(),
        ]
        llm.get_tool_calls.side_effect = [
            _mock_response_with_tools([("submit_verdict", verdict_args)]).choices[0].message.tool_calls,
            [],
        ]
        llm.usage = MagicMock(cost_usd=0.01, total_tokens=300)

        results = check_norms(session, llm, retriever)
        # Verdict rejected, param becomes MANUAL
        assert len(results) == 1
        assert results[0].status == "MANUAL"

    def test_low_confidence_becomes_manual(self):
        """submit_verdict with confidence < 0.7 -> status forced to MANUAL."""
        param = _make_parameter()
        session = _make_session(parameters=[param])
        retriever = _make_retriever()

        verdict_args = {
            "parameter_name": "Напряжение сети",
            "status": "FAIL",
            "norm_reference": "ПУЭ 7.1.34",
            "norm_requirement": "380В",
            "source_chunk_id": "pue_7_1_34_001",
            "confidence": 0.5,
            "explanation": "Не соответствует",
        }

        llm = MagicMock()
        llm.chat.side_effect = [
            _mock_response_with_tools([("submit_verdict", verdict_args)]),
            _mock_response_no_tools(),
        ]
        llm.get_tool_calls.side_effect = [
            _mock_response_with_tools([("submit_verdict", verdict_args)]).choices[0].message.tool_calls,
            [],
        ]
        llm.usage = MagicMock(cost_usd=0.01, total_tokens=300)

        results = check_norms(session, llm, retriever)
        assert len(results) == 1
        assert results[0].status == "MANUAL"
        assert "[low confidence" in results[0].explanation

    def test_step_limit_breaks_loop(self):
        """Agent hitting max_steps -> remaining params become MANUAL."""
        params = [_make_parameter(name=f"Param {i}") for i in range(5)]
        session = _make_session(parameters=params)
        retriever = _make_retriever()

        # Agent always calls search_norms, never submits verdict
        search_call = _mock_response_with_tools([
            ("search_norms", {"query": "test"})
        ])
        search_tool_calls = search_call.choices[0].message.tool_calls

        llm = MagicMock()
        llm.chat.return_value = search_call
        llm.get_tool_calls.return_value = search_tool_calls
        llm.usage = MagicMock(cost_usd=0.01, total_tokens=200)

        results = check_norms(session, llm, retriever)
        assert len(results) == 5
        assert all(r.status == "MANUAL" for r in results)
        # max_steps = min(3 * 5, 15) = 15
        assert llm.chat.call_count == 15

    def test_budget_exceeded_breaks_loop(self):
        """cost exceeding circuit_breaker -> loop breaks."""
        param = _make_parameter()
        session = _make_session(parameters=[param], cost_usd=0.99)
        retriever = _make_retriever()

        llm = MagicMock()
        # After first call, cost exceeds limit
        llm.usage = MagicMock(cost_usd=1.5, total_tokens=5000)
        llm.chat.return_value = _mock_response_with_tools([
            ("search_norms", {"query": "test"})
        ])
        llm.get_tool_calls.return_value = (
            _mock_response_with_tools([("search_norms", {"query": "test"})])
            .choices[0].message.tool_calls
        )

        results = check_norms(session, llm, retriever)
        assert len(results) == 1
        assert results[0].status == "MANUAL"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_checker.py -v -k "not TestCompareValues"`
Expected: FAIL — `ImportError: cannot import name 'check_norms' from 'backend.app.checker'`

- [ ] **Step 3: Implement checker.py**

Create `backend/app/checker.py`:

```python
"""Normative Checker: agentic RAG loop for parameter verification."""

from __future__ import annotations

import json
import logging

from backend.app.config import Settings
from backend.app.llm import LLMClient
from backend.app.prompts.checker import SYSTEM_PROMPT, make_user_prompt
from backend.app.retriever import NormRetriever
from backend.app.schemas import CheckResult, Parameter, SessionState
from backend.app.tools import (
    TOOL_SCHEMAS,
    execute_compare_values,
    execute_get_norm_chunk,
    execute_search_norms,
)

logger = logging.getLogger(__name__)


def _match_parameter(
    name: str,
    parameters: list[Parameter],
    used_names: set[str],
) -> Parameter | None:
    """Match parameter_name from verdict to a Parameter object.

    1. Exact match by Parameter.name
    2. Fuzzy: substring in either direction
    Skips already-used parameter names. Returns first unused match.
    """
    # Exact match
    for p in parameters:
        if p.name == name and p.name not in used_names:
            return p

    # Fuzzy: substring in either direction
    for p in parameters:
        if p.name in used_names:
            continue
        if name in p.name or p.name in name:
            return p

    return None


def _handle_submit_verdict(
    args: dict,
    parameters: list[Parameter],
    used_names: set[str],
    retriever: NormRetriever,
    settings: Settings,
) -> tuple[str, CheckResult | None]:
    """Process a submit_verdict tool call.

    Returns (tool_result_json, check_result_or_none).
    """
    param_name = args["parameter_name"]
    chunk_id = args["source_chunk_id"]
    confidence = args["confidence"]
    status = args["status"]
    explanation = args["explanation"]

    # Verify chunk_id exists
    if not retriever.get_norm_chunk(chunk_id):
        return (
            json.dumps(
                {"error": f"invalid chunk_id '{chunk_id}', use chunk IDs from search_norms results"},
                ensure_ascii=False,
            ),
            None,
        )

    # Match parameter
    param = _match_parameter(param_name, parameters, used_names)
    if param is None:
        return (
            json.dumps(
                {"error": f"parameter '{param_name}' not found in parameter list"},
                ensure_ascii=False,
            ),
            None,
        )

    # Confidence threshold
    if confidence < settings.confidence_threshold:
        status = "MANUAL"
        explanation = f"[low confidence -> MANUAL] {explanation}"

    used_names.add(param.name)

    check_result = CheckResult(
        parameter=param,
        status=status,
        norm_reference=args["norm_reference"],
        norm_requirement=args["norm_requirement"],
        source_chunk_id=chunk_id,
        confidence=confidence,
        explanation=explanation,
    )

    return json.dumps({"accepted": True}, ensure_ascii=False), check_result


def _execute_tool(
    tool_name: str,
    args: dict,
    retriever: NormRetriever,
) -> str:
    """Dispatch non-verdict tool calls."""
    if tool_name == "search_norms":
        return execute_search_norms(retriever, args)
    elif tool_name == "get_norm_chunk":
        return execute_get_norm_chunk(retriever, args)
    elif tool_name == "compare_values":
        return execute_compare_values(args)
    else:
        return json.dumps({"error": f"unknown tool: {tool_name}"}, ensure_ascii=False)


def check_norms(
    session: SessionState,
    llm: LLMClient,
    retriever: NormRetriever,
) -> list[CheckResult]:
    """Run the normative checker agent loop.

    Args:
        session: Session with populated parameters list.
        llm: LLM client for API calls.
        retriever: FAISS-based norm retriever.

    Returns:
        List of CheckResult for every parameter (PASS/FAIL/MANUAL).
    """
    if not session.parameters:
        return []

    settings = Settings()
    max_steps = min(3 * len(session.parameters), settings.max_agent_steps)

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": make_user_prompt(session.parameters)},
    ]

    check_results: list[CheckResult] = []
    used_names: set[str] = set()
    step = 0

    while step < max_steps:
        # Budget check
        if llm.usage.cost_usd >= settings.circuit_breaker_usd:
            logger.warning("Budget exceeded: $%.2f >= $%.2f", llm.usage.cost_usd, settings.circuit_breaker_usd)
            break

        response = llm.chat(
            messages,
            tools=TOOL_SCHEMAS,
            trace_name="checker-agent-step",
        )
        session.cost_usd = llm.usage.cost_usd
        session.token_usage = llm.usage.total_tokens

        tool_calls = llm.get_tool_calls(response)
        if not tool_calls:
            break

        # Append assistant message with tool calls
        assistant_msg = {"role": "assistant", "content": None, "tool_calls": []}
        for tc in tool_calls:
            assistant_msg["tool_calls"].append({
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            })
        messages.append(assistant_msg)

        # Execute each tool call
        for tc in tool_calls:
            tool_name = tc.function.name
            args = json.loads(tc.function.arguments)

            if tool_name == "submit_verdict":
                tool_result, check_result = _handle_submit_verdict(
                    args, session.parameters, used_names, retriever, settings,
                )
                if check_result:
                    check_results.append(check_result)
            else:
                tool_result = _execute_tool(tool_name, args, retriever)

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": tool_result,
            })

        step += 1
        session.agent_steps = step

    # Fill in MANUAL for unchecked parameters
    for p in session.parameters:
        if p.name not in used_names:
            check_results.append(
                CheckResult(
                    parameter=p,
                    status="MANUAL",
                    norm_reference="",
                    norm_requirement="",
                    source_chunk_id="",
                    confidence=0.0,
                    explanation="Агент не вынес вердикт в пределах лимита шагов",
                )
            )

    session.check_results = check_results
    return check_results
```

- [ ] **Step 4: Run all tests to verify they pass**

Run: `uv run pytest tests/test_checker.py -v`
Expected: all tests PASS

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check backend/app/checker.py tests/test_checker.py --fix
uv run ruff format backend/app/checker.py tests/test_checker.py
git add backend/app/checker.py tests/test_checker.py
git commit -m "feat(checker): implement agent loop with tool dispatch and anti-hallucination"
```

---

### Task 4: Integration Smoke Test

**Files:**
- Modify: `tests/test_checker.py`

- [ ] **Step 1: Write an integration test using search_norms tool dispatch**

Append to `tests/test_checker.py`:

```python
class TestToolDispatch:
    def test_search_norms_dispatches_to_retriever(self):
        """Agent calls search_norms -> retriever.search_norms is invoked."""
        param = _make_parameter()
        session = _make_session(parameters=[param])
        retriever = _make_retriever()

        search_args = {"query": "напряжение электроснабжение", "top_k": 5}
        verdict_args = {
            "parameter_name": "Напряжение сети",
            "status": "PASS",
            "norm_reference": "ПУЭ 7.1.34",
            "norm_requirement": "380В",
            "source_chunk_id": "pue_7_1_34_001",
            "confidence": 0.9,
            "explanation": "Соответствует",
        }

        resp1 = _mock_response_with_tools([("search_norms", search_args)])
        resp2 = _mock_response_with_tools([("submit_verdict", verdict_args)])
        resp3 = _mock_response_no_tools()

        llm = MagicMock()
        llm.chat.side_effect = [resp1, resp2, resp3]
        llm.get_tool_calls.side_effect = [
            resp1.choices[0].message.tool_calls,
            resp2.choices[0].message.tool_calls,
            [],
        ]
        llm.usage = MagicMock(cost_usd=0.02, total_tokens=500)

        results = check_norms(session, llm, retriever)

        retriever.search_norms.assert_called_once_with(
            "напряжение электроснабжение", top_k=5, filter_doc=None,
        )
        assert len(results) == 1
        assert results[0].status == "PASS"

    def test_get_norm_chunk_dispatches_to_retriever(self):
        """Agent calls get_norm_chunk -> retriever.get_norm_chunk is invoked."""
        param = _make_parameter()
        session = _make_session(parameters=[param])
        retriever = _make_retriever()

        chunk_args = {"chunk_id": "pue_7_1_34_001"}
        verdict_args = {
            "parameter_name": "Напряжение сети",
            "status": "PASS",
            "norm_reference": "ПУЭ 7.1.34",
            "norm_requirement": "380В",
            "source_chunk_id": "pue_7_1_34_001",
            "confidence": 0.9,
            "explanation": "Соответствует",
        }

        resp1 = _mock_response_with_tools([("get_norm_chunk", chunk_args)])
        resp2 = _mock_response_with_tools([("submit_verdict", verdict_args)])
        resp3 = _mock_response_no_tools()

        llm = MagicMock()
        llm.chat.side_effect = [resp1, resp2, resp3]
        llm.get_tool_calls.side_effect = [
            resp1.choices[0].message.tool_calls,
            resp2.choices[0].message.tool_calls,
            [],
        ]
        llm.usage = MagicMock(cost_usd=0.02, total_tokens=500)

        results = check_norms(session, llm, retriever)

        # get_norm_chunk called twice: once for the tool call, once for verdict validation
        assert retriever.get_norm_chunk.call_count == 2
        assert results[0].status == "PASS"
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `uv run pytest tests/test_checker.py::TestToolDispatch -v`
Expected: all tests PASS

- [ ] **Step 3: Run full test suite**

Run: `uv run pytest -v`
Expected: all existing tests + new tests PASS

- [ ] **Step 4: Lint and commit**

```bash
uv run ruff check tests/test_checker.py --fix
uv run ruff format tests/test_checker.py
git add tests/test_checker.py
git commit -m "test(checker): add tool dispatch integration tests"
```

---

### Task 5: Eval Test on Ground Truth

**Files:**
- Create: `tests/test_checker_eval.py`

- [ ] **Step 1: Create the eval test**

Create `tests/test_checker_eval.py`:

```python
"""Eval verification of Normative Checker on ground truth data.

Run: uv run pytest tests/test_checker_eval.py -v -s --run-eval
Requires: OPENROUTER_API_KEY in .env, FAISS index built
"""

import json
from pathlib import Path

import pytest

from backend.app.checker import check_norms
from backend.app.config import Settings
from backend.app.llm import LLMClient
from backend.app.parser import parse_document
from backend.app.extractor import extract_parameters
from backend.app.retriever import NormRetriever
from backend.app.schemas import Parameter, SessionState

SAMPLES_DIR = Path("data/samples")


def _load_retriever() -> NormRetriever:
    settings = Settings()
    return NormRetriever(
        index_path=settings.faiss_index_path,
        metadata_path=settings.metadata_path,
    )


def _violation_matches(violation: dict, check_results: list) -> bool:
    """Check if an expected violation was detected as FAIL."""
    param_name = violation["parameter_name"].lower()
    for cr in check_results:
        if cr.status != "FAIL":
            continue
        if param_name in cr.parameter.name.lower() or cr.parameter.name.lower() in param_name:
            return True
    return False


@pytest.mark.eval
class TestCheckerEval:
    @pytest.fixture(autouse=True)
    def skip_without_flag(self, request):
        if not request.config.getoption("--run-eval", default=False):
            pytest.skip("Eval tests require --run-eval flag")

    @pytest.fixture
    def llm(self):
        return LLMClient()

    @pytest.fixture
    def retriever(self):
        return _load_retriever()

    @pytest.mark.parametrize(
        "sample_dir",
        [d.name for d in sorted(SAMPLES_DIR.iterdir()) if d.is_dir()],
    )
    def test_checker_precision_recall(self, llm, retriever, sample_dir):
        sample_path = SAMPLES_DIR / sample_dir
        gt_path = sample_path / "ground_truth.json"
        gt = json.loads(gt_path.read_text())

        # Parse document
        pdfs = list(sample_path.glob("*.pdf"))
        assert pdfs, f"No PDF found in {sample_path}"
        raw_text, sections = parse_document(pdfs[0])

        # Extract parameters
        session = SessionState(
            session_id=f"eval-checker-{sample_dir}",
            sections=sections,
            raw_text=raw_text,
        )
        params = extract_parameters(session, llm)
        session.parameters = params

        print(f"\n{'='*60}")
        print(f"Document: {sample_dir}")
        print(f"Parameters extracted: {len(params)}")

        # Run checker
        results = check_norms(session, llm, retriever)

        # Measure recall: what fraction of expected violations were detected
        expected_violations = gt["expected_violations"]
        if expected_violations:
            found_violations = 0
            for ev in expected_violations:
                if _violation_matches(ev, results):
                    found_violations += 1
                    print(f"  HIT: {ev['parameter_name']}")
                else:
                    print(f"  MISS: {ev['parameter_name']}")
            recall = found_violations / len(expected_violations)
        else:
            recall = 1.0  # No violations expected

        # Measure precision: what fraction of FAIL verdicts are correct
        fail_results = [r for r in results if r.status == "FAIL"]
        if fail_results:
            correct_fails = 0
            for fr in fail_results:
                fr_name = fr.parameter.name.lower()
                for ev in expected_violations:
                    ev_name = ev["parameter_name"].lower()
                    if fr_name in ev_name or ev_name in fr_name:
                        correct_fails += 1
                        break
            precision = correct_fails / len(fail_results)
        else:
            precision = 1.0 if not expected_violations else 0.0

        print(f"\nResults: {len(results)} total, {len(fail_results)} FAIL, "
              f"{sum(1 for r in results if r.status == 'PASS')} PASS, "
              f"{sum(1 for r in results if r.status == 'MANUAL')} MANUAL")
        print(f"Recall: {recall:.0%} ({found_violations if expected_violations else 'N/A'}/{len(expected_violations)})")
        print(f"Precision: {precision:.0%} ({correct_fails if fail_results else 'N/A'}/{len(fail_results)})")
        print(f"Cost: ${session.cost_usd:.4f}, tokens: {session.token_usage}, agent steps: {session.agent_steps}")

        # Targets: recall >= 70%, precision >= 80%
        if expected_violations:
            assert recall >= 0.7, f"Recall {recall:.0%} < 70% for {sample_dir}"
        if fail_results:
            assert precision >= 0.8, f"Precision {precision:.0%} < 80% for {sample_dir}"
```

- [ ] **Step 2: Run eval test (requires API key and FAISS index)**

Run: `uv run pytest tests/test_checker_eval.py -v -s --run-eval`
Expected: tests run, precision/recall metrics printed. Targets: recall >= 70%, precision >= 80%.

- [ ] **Step 3: Lint and commit**

```bash
uv run ruff check tests/test_checker_eval.py --fix
uv run ruff format tests/test_checker_eval.py
git add tests/test_checker_eval.py
git commit -m "test(checker): add eval test for precision/recall on ground truth"
```

---

### Task 6: Final Verification

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest -v`
Expected: all tests PASS (eval tests skipped without --run-eval flag)

- [ ] **Step 2: Lint check**

Run: `uv run ruff check . && uv run ruff format --check .`
Expected: no errors

- [ ] **Step 3: Final commit if any changes needed**

```bash
git add -A
git commit -m "chore(checker): final cleanup for TASK_7"
```
