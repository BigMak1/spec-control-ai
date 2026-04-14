"""Normative Checker agent loop: agentic RAG for parameter compliance checking."""

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

_SUBMIT_VERDICT_REQUIRED_FIELDS = (
    "parameter_name",
    "status",
    "norm_reference",
    "norm_requirement",
    "source_chunk_id",
    "confidence",
    "explanation",
)


def _match_parameter(
    name: str,
    parameters: list[Parameter],
    used_names: set[str],
) -> Parameter | None:
    """Match a parameter name from the agent to an actual Parameter object.

    First tries exact match by Parameter.name (skipping already used names).
    Then tries fuzzy: substring in either direction (name in p.name or p.name in name),
    skipping used names.
    Returns first unused match, or None.
    """
    # Exact match
    for p in parameters:
        if p.name in used_names:
            continue
        if p.name == name:
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
    """Process a submit_verdict tool call from the agent.

    Returns a (json_response, optional CheckResult) tuple.
    """
    missing = [f for f in _SUBMIT_VERDICT_REQUIRED_FIELDS if f not in args]
    if missing:
        return (
            json.dumps(
                {"error": f"missing required fields: {', '.join(missing)}"},
                ensure_ascii=False,
            ),
            None,
        )

    chunk_id = args.get("source_chunk_id", "")
    chunk = retriever.get_norm_chunk(chunk_id)
    if chunk is None:
        return (
            json.dumps(
                {"error": f"source_chunk_id '{chunk_id}' not found in vector store"},
                ensure_ascii=False,
            ),
            None,
        )

    param_name = args.get("parameter_name", "")
    param = _match_parameter(param_name, parameters, used_names)
    if param is None:
        return (
            json.dumps(
                {"error": f"parameter '{param_name}' not found or already checked"},
                ensure_ascii=False,
            ),
            None,
        )

    status = args.get("status", "MANUAL")
    if status not in {"PASS", "FAIL", "MANUAL"}:
        return (
            json.dumps(
                {"error": f"invalid status '{status}', must be PASS, FAIL, or MANUAL"},
                ensure_ascii=False,
            ),
            None,
        )

    try:
        confidence = float(args.get("confidence", 0.0))
    except (ValueError, TypeError):
        return (
            json.dumps(
                {"error": "invalid confidence value, must be a number between 0.0 and 1.0"},
                ensure_ascii=False,
            ),
            None,
        )

    explanation = args.get("explanation", "")

    if confidence < settings.confidence_threshold:
        status = "MANUAL"
        explanation = f"[low confidence -> MANUAL] {explanation}"

    used_names.add(param.name)

    check_result = CheckResult(
        parameter=param,
        status=status,
        norm_reference=args.get("norm_reference", ""),
        norm_requirement=args.get("norm_requirement", ""),
        source_chunk_id=chunk_id,
        confidence=confidence,
        explanation=explanation,
    )

    return (
        json.dumps(
            {"status": "ok", "parameter": param.name, "verdict": status},
            ensure_ascii=False,
        ),
        check_result,
    )


def _execute_tool(
    tool_name: str,
    args: dict,
    retriever: NormRetriever,
) -> str:
    """Dispatch a tool call to the appropriate execution function."""
    if tool_name == "search_norms":
        return execute_search_norms(retriever, args)
    if tool_name == "get_norm_chunk":
        return execute_get_norm_chunk(retriever, args)
    if tool_name == "compare_values":
        return execute_compare_values(args)
    return json.dumps(
        {"error": f"unknown tool: '{tool_name}'"},
        ensure_ascii=False,
    )


def check_norms(
    session: SessionState,
    llm: LLMClient,
    retriever: NormRetriever,
) -> list[CheckResult]:
    """Run the Normative Checker agent loop.

    Iterates through parameters, calling LLM with tools to search norms,
    compare values, and submit verdicts. Enforces step and budget limits.

    Budget is checked at the start of each step (before llm.chat). Because a
    step can only be interrupted between LLM calls — not mid-request — the
    final ``session.cost_usd`` may overshoot ``circuit_breaker_usd`` by up to
    one step's cost. This is an intentional trade-off: cheaper and simpler
    than mid-request cancellation, bounded by per-step token limits.
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
            logger.warning(
                "Budget exceeded: $%.4f >= $%.2f, stopping agent",
                llm.usage.cost_usd,
                settings.circuit_breaker_usd,
            )
            break

        response = llm.chat(
            messages,
            tools=TOOL_SCHEMAS,
            trace_name="checker-agent-step",
        )

        # Update session cost and token usage
        session.cost_usd = llm.usage.cost_usd
        session.token_usage = llm.usage.total_tokens

        tool_calls = llm.get_tool_calls(response)
        if not tool_calls:
            break

        # Append assistant message with tool_calls to messages
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            }
        )

        # Process each tool call
        for tc in tool_calls:
            tool_name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}

            if tool_name == "submit_verdict":
                tool_result, check_result = _handle_submit_verdict(
                    args, session.parameters, used_names, retriever, settings
                )
                if check_result is not None:
                    check_results.append(check_result)
            else:
                tool_result = _execute_tool(tool_name, args, retriever)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_result,
                }
            )

        step += 1
        session.agent_steps = step

    # Unchecked parameters become MANUAL
    for param in session.parameters:
        if param.name not in used_names:
            check_results.append(
                CheckResult(
                    parameter=param,
                    status="MANUAL",
                    confidence=0.0,
                    explanation="Агент не вынес вердикт в пределах лимита шагов",
                    source_chunk_id="",
                    norm_reference="",
                    norm_requirement="",
                )
            )

    session.check_results = check_results
    return check_results
