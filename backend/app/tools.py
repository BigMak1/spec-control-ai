"""Tool schemas and execution functions for the Normative Checker agent."""

from __future__ import annotations

import json
import re

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_norms",
            "description": (
                "Semantic search over normative documents (ГОСТы, СНиПы, СП, ПУЭ). "
                "Returns ranked chunks with metadata."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Search query in Russian describing the normative requirement to find."
                        ),
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Maximum number of chunks to return.",
                        "default": 5,
                    },
                    "filter_doc": {
                        "type": "string",
                        "description": (
                            "Optional document name filter (e.g. 'ПУЭ', 'СП 60.13330'). "
                            "Only chunks from this document are returned."
                        ),
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
            "description": (
                "Retrieve a specific normative chunk by its ID, "
                "including its neighbouring chunks for context."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "chunk_id": {
                        "type": "string",
                        "description": "Unique chunk identifier returned by search_norms.",
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
            "description": (
                "Deterministically compare an actual value from the document "
                "against a required normative value."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "actual": {
                        "type": "string",
                        "description": "The actual value extracted from the document.",
                    },
                    "required": {
                        "type": "string",
                        "description": "The required normative value.",
                    },
                    "comparison_type": {
                        "type": "string",
                        "enum": ["gte", "lte", "eq", "contains"],
                        "description": (
                            "Comparison operator: "
                            "'gte' (actual >= required), "
                            "'lte' (actual <= required), "
                            "'eq' (actual == required), "
                            "'contains' (required substring in actual)."
                        ),
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
            "description": (
                "Submit the final compliance verdict for a parameter. "
                "Must be called once per parameter after gathering all evidence."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "parameter_name": {
                        "type": "string",
                        "description": "Name of the checked parameter.",
                    },
                    "status": {
                        "type": "string",
                        "enum": ["PASS", "FAIL", "MANUAL"],
                        "description": (
                            "'PASS' — compliant, 'FAIL' — non-compliant, "
                            "'MANUAL' — requires human review (confidence < 0.7)."
                        ),
                    },
                    "norm_reference": {
                        "type": "string",
                        "description": (
                            "Citation of the normative document and clause (e.g. 'ПУЭ п.1.3.6')."
                        ),
                    },
                    "norm_requirement": {
                        "type": "string",
                        "description": "Verbatim or paraphrased normative requirement text.",
                    },
                    "source_chunk_id": {
                        "type": "string",
                        "description": "chunk_id from the vector store that backs this verdict.",
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Confidence score in [0.0, 1.0].",
                    },
                    "explanation": {
                        "type": "string",
                        "description": "Human-readable explanation of the verdict.",
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


def _parse_number(s: str) -> float | None:
    """Extract the first number from a string like '2.5 мм²' → 2.5."""
    match = re.search(r"[-+]?\d+(?:[.,]\d+)?", s)
    if match is None:
        return None
    return float(match.group().replace(",", "."))


def compare_values(actual: str, required: str, comparison_type: str) -> dict:
    """Deterministically compare actual vs required value.

    Returns a dict with keys:
      match: bool | None  — None if values cannot be parsed numerically
      actual_parsed: float | str | None
      required_parsed: float | str | None
      explanation: str
    """
    if comparison_type == "contains":
        # Use word-boundary-aware check: required must not be preceded by an alphanumeric char.
        # This prevents "ВВГнг" matching inside "АВВГнг".
        pattern = r"(?<![A-Za-zА-Яа-яЁёA-Za-z0-9])" + re.escape(required)
        match = bool(re.search(pattern, actual))
        return {
            "match": match,
            "actual_parsed": actual,
            "required_parsed": required,
            "explanation": (f"'{required}' {'found' if match else 'not found'} in '{actual}'"),
        }

    # Numeric comparisons: gte, lte, eq
    actual_num = _parse_number(actual)
    required_num = _parse_number(required)

    if actual_num is None or required_num is None:
        unparseable = []
        if actual_num is None:
            unparseable.append(f"actual='{actual}'")
        if required_num is None:
            unparseable.append(f"required='{required}'")
        return {
            "match": None,
            "actual_parsed": actual_num,
            "required_parsed": required_num,
            "explanation": f"cannot parse numerically: {', '.join(unparseable)}",
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
            "actual_parsed": actual_num,
            "required_parsed": required_num,
            "explanation": f"unknown comparison_type: '{comparison_type}'",
        }

    return {
        "match": match,
        "actual_parsed": actual_num,
        "required_parsed": required_num,
        "explanation": (
            f"{actual_num} {comparison_type} {required_num} → {'PASS' if match else 'FAIL'}"
        ),
    }


def execute_search_norms(retriever, args: dict) -> str:
    """Call retriever.search_norms() and return JSON string."""
    query = args["query"]
    top_k = args.get("top_k", 5)
    filter_doc = args.get("filter_doc")

    results = retriever.search_norms(query, top_k=top_k, filter_doc=filter_doc)

    # Truncate chunk text to 300 chars to save tokens
    truncated = []
    for item in results:
        entry = dict(item)
        if "metadata" in entry and "text" in entry["metadata"]:
            meta = dict(entry["metadata"])
            meta["text"] = meta["text"][:300]
            entry["metadata"] = meta
        truncated.append(entry)

    return json.dumps(truncated, ensure_ascii=False)


def execute_get_norm_chunk(retriever, args: dict) -> str:
    """Call retriever.get_norm_chunk() and return JSON string."""
    chunk_id = args["chunk_id"]
    result = retriever.get_norm_chunk(chunk_id)
    if result is None:
        return json.dumps({"error": f"chunk '{chunk_id}' not found"}, ensure_ascii=False)
    return json.dumps(result, ensure_ascii=False)


def execute_compare_values(args: dict) -> str:
    """Call compare_values() and return JSON string."""
    result = compare_values(
        actual=args["actual"],
        required=args["required"],
        comparison_type=args["comparison_type"],
    )
    return json.dumps(result, ensure_ascii=False)
