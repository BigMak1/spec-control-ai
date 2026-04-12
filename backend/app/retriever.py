"""FAISS-based normative document retriever."""

from __future__ import annotations

import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
QUERY_PREFIX = "query: "
MIN_SCORE = 0.3


class NormRetriever:
    """Search normative chunks using FAISS index."""

    def __init__(
        self,
        index_path: str,
        metadata_path: str,
        min_score: float = MIN_SCORE,
    ) -> None:
        self.index = faiss.read_index(index_path)
        with open(metadata_path, encoding="utf-8") as f:
            self.metadata: list[dict] = json.load(f)

        self._id_to_pos: dict[str, int] = {m["chunk_id"]: i for i, m in enumerate(self.metadata)}

        self._model: SentenceTransformer | None = None
        self._min_score = min_score

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(EMBEDDING_MODEL)
        return self._model

    def search_norms(
        self,
        query: str,
        top_k: int = 5,
        filter_doc: str | None = None,
    ) -> list[dict]:
        """Search normative chunks by semantic similarity."""
        model = self._get_model()
        query_embedding = model.encode(
            QUERY_PREFIX + query,
            normalize_embeddings=True,
        )
        query_vec = np.array([query_embedding], dtype=np.float32)

        search_k = top_k * 3 if filter_doc else top_k
        scores, indices = self.index.search(query_vec, min(search_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or score < self._min_score:
                continue
            chunk_meta = self.metadata[idx]
            if filter_doc and filter_doc not in chunk_meta["norm_doc"]:
                continue
            results.append({"metadata": chunk_meta, "score": float(score)})
            if len(results) >= top_k:
                break

        return results

    def get_norm_chunk(self, chunk_id: str) -> dict | None:
        """Get a chunk by ID with its prev/next neighbors."""
        pos = self._id_to_pos.get(chunk_id)
        if pos is None:
            return None

        chunk = self.metadata[pos]
        prev_chunk = self.metadata[pos - 1] if pos > 0 else None
        next_chunk = self.metadata[pos + 1] if pos + 1 < len(self.metadata) else None

        return {
            "chunk": chunk,
            "prev": prev_chunk,
            "next": next_chunk,
        }
