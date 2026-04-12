from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class Section(BaseModel):
    name: str
    text: str
    page_start: int
    page_end: int


class Parameter(BaseModel):
    name: str
    value: str
    unit: str | None = None
    context: str
    source_page: int
    source_text: str


class CheckResult(BaseModel):
    parameter: Parameter
    status: Literal["PASS", "FAIL", "MANUAL"]
    norm_reference: str
    norm_requirement: str
    source_chunk_id: str
    confidence: float
    explanation: str


class SessionState(BaseModel):
    session_id: str
    status: Literal["parsing", "extracting", "checking", "reporting", "done", "error"] = "parsing"
    raw_text: str = ""
    anonymized_text: str = ""
    pii_map: dict = {}
    sections: list[Section] = []
    parameters: list[Parameter] = []
    check_results: list[CheckResult] = []
    report: str = ""
    token_usage: int = 0
    cost_usd: float = 0.0
    agent_steps: int = 0


class ChunkMetadata(BaseModel):
    chunk_id: str
    norm_doc: str
    section: str
    title: str
    page: int
    text: str
    version: str
    status: str


class ChunkResult(BaseModel):
    metadata: ChunkMetadata
    score: float
