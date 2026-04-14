"""Eval verification of Normative Checker on ground truth data.

Run: uv run pytest tests/test_checker_eval.py -v -s --run-eval
Requires: OPENROUTER_API_KEY in .env, FAISS index built
"""

import json
from pathlib import Path

import pytest

from backend.app.checker import check_norms
from backend.app.config import Settings
from backend.app.extractor import extract_parameters
from backend.app.llm import LLMClient
from backend.app.parser import parse_document
from backend.app.retriever import NormRetriever
from backend.app.schemas import SessionState

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

        print(f"\n{'=' * 60}")
        print(f"Document: {sample_dir}")
        print(f"Parameters extracted: {len(params)}")

        # Run checker
        results = check_norms(session, llm, retriever)

        # Measure recall: what fraction of expected violations were detected
        expected_violations = gt["expected_violations"]
        found_violations = 0
        if expected_violations:
            for ev in expected_violations:
                if _violation_matches(ev, results):
                    found_violations += 1
                    print(f"  HIT: {ev['parameter_name']}")
                else:
                    print(f"  MISS: {ev['parameter_name']}")
            recall = found_violations / len(expected_violations)
        else:
            recall = 1.0

        # Measure precision: what fraction of FAIL verdicts match expected violations
        fail_results = [r for r in results if r.status == "FAIL"]
        correct_fails = 0
        if fail_results:
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

        pass_count = sum(1 for r in results if r.status == "PASS")
        manual_count = sum(1 for r in results if r.status == "MANUAL")
        print(
            f"\nResults: {len(results)} total, {len(fail_results)} FAIL, "
            f"{pass_count} PASS, {manual_count} MANUAL"
        )
        if expected_violations:
            print(f"Recall: {recall:.0%} ({found_violations}/{len(expected_violations)})")
        if fail_results:
            print(f"Precision: {precision:.0%} ({correct_fails}/{len(fail_results)})")
        print(
            f"Cost: ${session.cost_usd:.4f}, tokens: {session.token_usage}, "
            f"agent steps: {session.agent_steps}"
        )

        # Targets: recall >= 70%, precision >= 80%
        if expected_violations:
            assert recall >= 0.7, f"Recall {recall:.0%} < 70% for {sample_dir}"
        if fail_results:
            assert precision >= 0.8, f"Precision {precision:.0%} < 80% for {sample_dir}"
