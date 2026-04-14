import json
from unittest.mock import MagicMock

from backend.app.checker import _match_parameter, check_norms
from backend.app.schemas import Parameter, SessionState
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

    def test_comma_decimal_separator(self):
        result = compare_values("2,5 мм²", "2.5", "gte")
        assert result["match"] is True
        assert result["actual_parsed"] == 2.5


# ---------------------------------------------------------------------------
# Helpers for checker tests
# ---------------------------------------------------------------------------


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


def _mock_response_no_tools(content="Готово."):
    resp = MagicMock()
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = None
    resp.choices = [MagicMock(message=msg, finish_reason="stop")]
    resp.usage = MagicMock(prompt_tokens=100, completion_tokens=50)
    return resp


def _mock_response_with_tools(tool_calls_data: list[tuple[str, dict]]):
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


# ---------------------------------------------------------------------------
# TestMatchParameter
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# TestCheckNorms
# ---------------------------------------------------------------------------


class TestCheckNorms:
    def test_no_parameters(self):
        session = _make_session(parameters=[])
        llm = MagicMock()
        retriever = _make_retriever()
        results = check_norms(session, llm, retriever)
        assert results == []
        llm.chat.assert_not_called()

    def test_agent_stops_immediately(self):
        """LLM returns no tool_calls -> all params become MANUAL."""
        param = _make_parameter()
        session = _make_session(parameters=[param])
        llm = MagicMock()
        resp = _mock_response_no_tools()
        llm.chat.return_value = resp
        llm.get_tool_calls.return_value = []
        llm.usage = MagicMock(cost_usd=0.001, total_tokens=150)
        retriever = _make_retriever()

        results = check_norms(session, llm, retriever)
        assert len(results) == 1
        assert results[0].status == "MANUAL"
        assert "не вынес вердикт" in results[0].explanation

    def test_submit_verdict_saves_result(self):
        """LLM calls submit_verdict with valid args -> CheckResult saved."""
        param = _make_parameter(name="Напряжение сети")
        session = _make_session(parameters=[param])

        verdict_args = {
            "parameter_name": "Напряжение сети",
            "status": "PASS",
            "norm_reference": "ПУЭ 7.1.34",
            "norm_requirement": "Напряжение сети должно быть 380В",
            "source_chunk_id": "pue_7_1_34_001",
            "confidence": 0.9,
            "explanation": "Параметр соответствует нормативу",
        }

        resp_tools = _mock_response_with_tools([("submit_verdict", verdict_args)])
        resp_stop = _mock_response_no_tools()

        llm = MagicMock()
        llm.chat.side_effect = [resp_tools, resp_stop]
        llm.get_tool_calls.side_effect = [
            resp_tools.choices[0].message.tool_calls,
            [],
        ]
        llm.usage = MagicMock(cost_usd=0.002, total_tokens=300)

        retriever = _make_retriever()
        results = check_norms(session, llm, retriever)

        assert len(results) == 1
        assert results[0].status == "PASS"
        assert results[0].source_chunk_id == "pue_7_1_34_001"
        assert results[0].parameter.name == "Напряжение сети"

    def test_invalid_chunk_id_rejected(self):
        """retriever.get_norm_chunk returns None -> verdict rejected, param MANUAL."""
        param = _make_parameter(name="Напряжение сети")
        session = _make_session(parameters=[param])

        verdict_args = {
            "parameter_name": "Напряжение сети",
            "status": "PASS",
            "norm_reference": "ПУЭ 7.1.34",
            "norm_requirement": "Напряжение сети должно быть 380В",
            "source_chunk_id": "nonexistent_chunk",
            "confidence": 0.9,
            "explanation": "Параметр соответствует нормативу",
        }

        resp_tools = _mock_response_with_tools([("submit_verdict", verdict_args)])
        resp_stop = _mock_response_no_tools()

        llm = MagicMock()
        llm.chat.side_effect = [resp_tools, resp_stop]
        llm.get_tool_calls.side_effect = [
            resp_tools.choices[0].message.tool_calls,
            [],
        ]
        llm.usage = MagicMock(cost_usd=0.002, total_tokens=300)

        retriever = _make_retriever()
        # Override get_norm_chunk to return None for the bad chunk_id
        retriever.get_norm_chunk.return_value = None

        results = check_norms(session, llm, retriever)
        assert len(results) == 1
        assert results[0].status == "MANUAL"
        assert "не вынес вердикт" in results[0].explanation

    def test_low_confidence_becomes_manual(self):
        """confidence=0.5, status='FAIL' -> forced to MANUAL."""
        param = _make_parameter(name="Напряжение сети")
        session = _make_session(parameters=[param])

        verdict_args = {
            "parameter_name": "Напряжение сети",
            "status": "FAIL",
            "norm_reference": "ПУЭ 7.1.34",
            "norm_requirement": "Напряжение сети должно быть 380В",
            "source_chunk_id": "pue_7_1_34_001",
            "confidence": 0.5,
            "explanation": "Параметр не соответствует нормативу",
        }

        resp_tools = _mock_response_with_tools([("submit_verdict", verdict_args)])
        resp_stop = _mock_response_no_tools()

        llm = MagicMock()
        llm.chat.side_effect = [resp_tools, resp_stop]
        llm.get_tool_calls.side_effect = [
            resp_tools.choices[0].message.tool_calls,
            [],
        ]
        llm.usage = MagicMock(cost_usd=0.002, total_tokens=300)

        retriever = _make_retriever()
        results = check_norms(session, llm, retriever)

        assert len(results) == 1
        assert results[0].status == "MANUAL"
        assert "[low confidence" in results[0].explanation

    def test_step_limit_breaks_loop(self):
        """5 params, agent always calls search_norms -> loop breaks at max_steps."""
        params = [_make_parameter(name=f"Param {i}") for i in range(5)]
        session = _make_session(parameters=params)

        # Agent always calls search_norms, never submits
        search_resp = _mock_response_with_tools([("search_norms", {"query": "test query"})])

        llm = MagicMock()
        # max_steps = min(3*5, 15) = 15, so we need 15 responses + won't reach 16th
        llm.chat.side_effect = [search_resp] * 16
        llm.get_tool_calls.side_effect = [search_resp.choices[0].message.tool_calls] * 16
        llm.usage = MagicMock(cost_usd=0.01, total_tokens=1000)

        retriever = _make_retriever()
        results = check_norms(session, llm, retriever)

        # All 5 params should be MANUAL since none got a verdict
        assert len(results) == 5
        assert all(r.status == "MANUAL" for r in results)
        assert all("не вынес вердикт" in r.explanation for r in results)
        # Agent should have done exactly 15 steps
        assert session.agent_steps == 15

    def test_budget_exceeded_breaks_loop(self):
        """llm.usage.cost_usd=1.5 after first call -> loop breaks."""
        param = _make_parameter()
        session = _make_session(parameters=[param])

        llm = MagicMock()
        # cost already exceeds circuit_breaker_usd (1.0) before first call
        llm.usage = MagicMock(cost_usd=1.5, total_tokens=5000)

        retriever = _make_retriever()
        results = check_norms(session, llm, retriever)

        # Should not even call llm.chat since budget is exceeded
        llm.chat.assert_not_called()
        assert len(results) == 1
        assert results[0].status == "MANUAL"

    def test_mixed_verdicts_and_manual_fallback(self):
        """2 params, agent submits verdict for one, step limit hit for other."""
        p1 = _make_parameter(name="Напряжение сети", value="380", unit="В")
        p2 = _make_parameter(name="Сечение кабеля", value="2.5", unit="мм²")
        session = _make_session(parameters=[p1, p2])
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
        search_args = {"query": "сечение кабеля"}

        resp_verdict = _mock_response_with_tools([("submit_verdict", verdict_args)])
        resp_search = _mock_response_with_tools([("search_norms", search_args)])

        llm = MagicMock()
        # First call: submit verdict for p1, then loop forever searching for p2 until max_steps
        llm.chat.side_effect = [resp_verdict] + [resp_search] * 20
        llm.get_tool_calls.side_effect = [
            resp_verdict.choices[0].message.tool_calls,
        ] + [resp_search.choices[0].message.tool_calls] * 20
        llm.usage = MagicMock(cost_usd=0.01, total_tokens=300)

        results = check_norms(session, llm, retriever)
        assert len(results) == 2
        statuses = [r.status for r in results]
        assert "PASS" in statuses
        assert "MANUAL" in statuses
        # PASS result should be for p1
        pass_result = next(r for r in results if r.status == "PASS")
        assert pass_result.parameter.name == "Напряжение сети"


# ---------------------------------------------------------------------------
# TestToolDispatch
# ---------------------------------------------------------------------------


class TestToolDispatch:
    def test_search_norms_dispatches_to_retriever(self):
        """Agent calls search_norms -> retriever.search_norms is invoked with correct args."""
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
            "напряжение электроснабжение",
            top_k=5,
            filter_doc=None,
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

        # get_norm_chunk called: once for the tool call, once for verdict validation
        assert retriever.get_norm_chunk.call_count == 2
        assert results[0].status == "PASS"
