"""Eval-верификация Parameter Extractor на реальных документах.

Запуск: uv run pytest tests/test_extractor_eval.py -v -s --run-eval
Требует: OPENROUTER_API_KEY в .env
"""

import json
from pathlib import Path

import pytest

from backend.app.extractor import extract_parameters
from backend.app.llm import LLMClient
from backend.app.parser import parse_document
from backend.app.schemas import SessionState

SAMPLES_DIR = Path("data/samples")


def _fuzzy_match(expected_name: str, expected_value: str, params: list) -> bool:
    """Нечёткое сравнение: ищем параметр с похожим name и совпадающим value."""
    exp_name_lower = expected_name.lower()
    for p in params:
        if expected_value in p.value and (
            exp_name_lower in p.name.lower()
            or any(word in p.name.lower() for word in exp_name_lower.split() if len(word) > 3)
        ):
            return True
    return False


@pytest.mark.eval
class TestExtractorEval:
    @pytest.fixture(autouse=True)
    def skip_without_flag(self, request):
        if not request.config.getoption("--run-eval", default=False):
            pytest.skip("Eval tests require --run-eval flag")

    @pytest.fixture
    def llm(self):
        return LLMClient()

    @pytest.mark.parametrize(
        "sample_dir",
        [d.name for d in sorted(SAMPLES_DIR.iterdir()) if d.is_dir()],
    )
    def test_recall(self, llm, sample_dir):
        sample_path = SAMPLES_DIR / sample_dir
        gt_path = sample_path / "ground_truth.json"
        gt = json.loads(gt_path.read_text())

        # Ищем PDF
        pdfs = list(sample_path.glob("*.pdf"))
        assert pdfs, f"No PDF found in {sample_path}"
        pdf_path = pdfs[0]

        # Parse
        raw_text, sections = parse_document(pdf_path)

        # Extract
        session = SessionState(
            session_id=f"eval-{sample_dir}",
            sections=sections,
            raw_text=raw_text,
        )
        params = extract_parameters(session, llm)

        # Measure recall
        expected = gt["expected_parameters"]
        found = 0
        for ep in expected:
            if _fuzzy_match(ep["name"], ep["value"], params):
                found += 1
            else:
                print(f"  MISS: {ep['name']} = {ep['value']}")

        recall = found / len(expected) if expected else 1.0
        print(f"\n{sample_dir}: {found}/{len(expected)} recall={recall:.0%}")
        print(f"  Total params extracted: {len(params)}")
        print(f"  Cost: ${session.cost_usd:.4f}, tokens: {session.token_usage}")

        # Цель: recall >= 70%
        assert recall >= 0.7, f"Recall {recall:.0%} < 70% for {sample_dir}"
