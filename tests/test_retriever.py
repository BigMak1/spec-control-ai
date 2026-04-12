import json
from pathlib import Path

import faiss
import numpy as np

from backend.app.retriever import NormRetriever


def _create_test_index(tmp_path: Path) -> tuple[Path, Path]:
    """Create a small FAISS index and metadata for testing."""
    dim = 1024
    n_chunks = 5

    rng = np.random.default_rng(42)
    vectors = rng.standard_normal((n_chunks, dim)).astype(np.float32)
    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    metadata = [
        {
            "chunk_id": f"pue_7_1_{30 + i}_001",
            "norm_doc": "ПУЭ 7-е изд.",
            "section": f"7.1.{30 + i}",
            "title": f"Пункт {30 + i}",
            "page": 140 + i,
            "text": f"Текст пункта 7.1.{30 + i} о требованиях.",
            "version": "2003",
            "status": "действующий",
        }
        for i in range(n_chunks)
    ]

    index_path = tmp_path / "index.faiss"
    metadata_path = tmp_path / "metadata.json"
    faiss.write_index(index, str(index_path))
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False)

    return index_path, metadata_path


def test_retriever_loads_index(tmp_path):
    index_path, metadata_path = _create_test_index(tmp_path)
    retriever = NormRetriever(str(index_path), str(metadata_path))
    assert retriever.index.ntotal == 5
    assert len(retriever.metadata) == 5


def test_search_norms_returns_results(tmp_path):
    index_path, metadata_path = _create_test_index(tmp_path)
    retriever = NormRetriever(str(index_path), str(metadata_path), min_score=0.0)
    results = retriever.search_norms("сечение кабеля", top_k=3)
    assert len(results) <= 3
    for r in results:
        assert "metadata" in r
        assert "score" in r
        assert r["metadata"]["norm_doc"] == "ПУЭ 7-е изд."


def test_search_norms_filter_doc(tmp_path):
    index_path, metadata_path = _create_test_index(tmp_path)
    retriever = NormRetriever(str(index_path), str(metadata_path), min_score=0.0)
    results = retriever.search_norms("текст", top_k=5, filter_doc="СП 60")
    # All chunks are "ПУЭ 7-е изд.", so filtering by "СП 60" should return empty
    assert len(results) == 0


def test_get_norm_chunk_found(tmp_path):
    index_path, metadata_path = _create_test_index(tmp_path)
    retriever = NormRetriever(str(index_path), str(metadata_path))
    result = retriever.get_norm_chunk("pue_7_1_32_001")
    assert result is not None
    assert result["chunk"]["chunk_id"] == "pue_7_1_32_001"
    assert result["prev"]["chunk_id"] == "pue_7_1_31_001"
    assert result["next"]["chunk_id"] == "pue_7_1_33_001"


def test_get_norm_chunk_first_has_no_prev(tmp_path):
    index_path, metadata_path = _create_test_index(tmp_path)
    retriever = NormRetriever(str(index_path), str(metadata_path))
    result = retriever.get_norm_chunk("pue_7_1_30_001")
    assert result is not None
    assert result["prev"] is None
    assert result["next"] is not None


def test_get_norm_chunk_not_found(tmp_path):
    index_path, metadata_path = _create_test_index(tmp_path)
    retriever = NormRetriever(str(index_path), str(metadata_path))
    result = retriever.get_norm_chunk("nonexistent_id")
    assert result is None
