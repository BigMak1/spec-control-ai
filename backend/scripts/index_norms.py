"""Offline indexation of normative PDFs into FAISS vector store."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import faiss
import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer

# --- Chunking logic ---

CHARS_PER_TOKEN = 4
MAX_CHUNK_TOKENS = 800
OVERLAP_TOKENS = 100
FALLBACK_CHUNK_TOKENS = 600
MIN_SECTIONS_FOR_SEMANTIC = 1

_SECTION_RE = re.compile(
    r"^(\d+\.\d+(?:\.\d+)*)\s+",
    re.MULTILINE,
)


def detect_sections(text: str) -> list[dict]:
    """Detect numbered sections in text using regex.

    Returns list of dicts: {"section": "7.1.34", "text": "full section text"}.
    """
    matches = list(_SECTION_RE.finditer(text))
    if not matches:
        return []

    sections = []
    for i, match in enumerate(matches):
        section_id = match.group(1)
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        sections.append({"section": section_id, "text": section_text})

    return sections


def _estimate_tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN


def _split_long_text(text: str, title_prefix: str) -> list[str]:
    """Split long text into chunks of ~MAX_CHUNK_TOKENS with overlap.

    The title_prefix is prepended to every sub-chunk so that each chunk
    carries the section identifier for retrieval context.
    """
    max_chars = MAX_CHUNK_TOKENS * CHARS_PER_TOKEN
    overlap_chars = OVERLAP_TOKENS * CHARS_PER_TOKEN
    prefix = (title_prefix + " ") if title_prefix else ""
    prefix_chars = len(prefix)
    body_max = max_chars - prefix_chars
    chunks = []
    start = 0
    while start < len(text):
        end = start + body_max
        body = text[start:end]
        chunks.append((prefix + body).strip())
        start = end - overlap_chars
    return chunks


def _make_chunk_id(doc_slug: str, section: str, seq: int) -> str:
    section_part = section.replace(".", "_") if section else "nosec"
    return f"{doc_slug}_{section_part}_{seq:03d}"


def chunk_text(
    text: str,
    doc_slug: str,
    norm_doc: str,
    version: str,
    default_page: int = 0,
) -> list[dict]:
    """Split text into chunks using hybrid strategy.

    1. Try regex section detection
    2. If enough sections found: use semantic chunking by sections
    3. If section is too long: split with overlap
    4. Fallback: fixed-size chunking
    """
    sections = detect_sections(text)

    if len(sections) >= MIN_SECTIONS_FOR_SEMANTIC:
        return _chunk_by_sections(sections, doc_slug, norm_doc, version, default_page)
    else:
        return _chunk_fixed_size(text, doc_slug, norm_doc, version, default_page)


def _chunk_by_sections(
    sections: list[dict],
    doc_slug: str,
    norm_doc: str,
    version: str,
    default_page: int,
) -> list[dict]:
    chunks = []
    for sec in sections:
        section_id = sec["section"]
        section_text = sec["text"]
        first_line = section_text.split("\n")[0].strip()
        title_prefix = first_line[:120] if first_line else section_id

        if _estimate_tokens(section_text) > MAX_CHUNK_TOKENS:
            # Pass the body (everything after the first line) to avoid
            # duplicating the header; title_prefix is prepended to every chunk.
            lines = section_text.split("\n", 1)
            body = lines[1] if len(lines) > 1 else lines[0]
            sub_texts = _split_long_text(body, title_prefix)
            for i, sub_text in enumerate(sub_texts):
                chunks.append(
                    {
                        "chunk_id": _make_chunk_id(doc_slug, section_id, i + 1),
                        "norm_doc": norm_doc,
                        "section": section_id,
                        "title": title_prefix,
                        "page": default_page,
                        "text": sub_text,
                        "version": version,
                        "status": "действующий",
                    }
                )
        else:
            chunks.append(
                {
                    "chunk_id": _make_chunk_id(doc_slug, section_id, 1),
                    "norm_doc": norm_doc,
                    "section": section_id,
                    "title": title_prefix,
                    "page": default_page,
                    "text": section_text,
                    "version": version,
                    "status": "действующий",
                }
            )
    return chunks


def _chunk_fixed_size(
    text: str,
    doc_slug: str,
    norm_doc: str,
    version: str,
    default_page: int,
) -> list[dict]:
    max_chars = FALLBACK_CHUNK_TOKENS * CHARS_PER_TOKEN
    overlap_chars = OVERLAP_TOKENS * CHARS_PER_TOKEN
    chunks = []
    start = 0
    seq = 1
    while start < len(text):
        end = start + max_chars
        chunk_text_str = text[start:end].strip()
        if chunk_text_str:
            chunks.append(
                {
                    "chunk_id": _make_chunk_id(doc_slug, "", seq),
                    "norm_doc": norm_doc,
                    "section": "",
                    "title": "",
                    "page": default_page,
                    "text": chunk_text_str,
                    "version": version,
                    "status": "действующий",
                }
            )
            seq += 1
        start = end - overlap_chars
    return chunks


# --- PDF reading ---


def extract_text_from_pdf(pdf_path: str | Path) -> str:
    """Extract full text from a PDF file using PyMuPDF."""
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        pages.append(page.get_text())
    doc.close()
    return "\n".join(pages)


# --- Config for known normative documents ---

NORM_DOCS = {
    "pue_7.pdf": {
        "doc_slug": "pue",
        "norm_doc": "ПУЭ 7-е изд.",
        "version": "2003",
    },
    "sp_60_13330_2020.pdf": {
        "doc_slug": "sp60",
        "norm_doc": "СП 60.13330.2020",
        "version": "2020",
    },
}

EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
PASSAGE_PREFIX = "passage: "
EMBEDDING_DIM = 1024


# --- Main indexing ---


def build_index(norms_dir: str | Path, output_dir: str | Path) -> None:
    """Read all normative PDFs, chunk, embed, and save FAISS index + metadata."""
    norms_dir = Path(norms_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    all_chunks: list[dict] = []

    for pdf_file in sorted(norms_dir.glob("*.pdf")):
        doc_config = NORM_DOCS.get(pdf_file.name)
        if doc_config is None:
            print(f"  Skipping {pdf_file.name}: not in NORM_DOCS config")
            continue

        print(f"Processing: {pdf_file.name}")
        text = extract_text_from_pdf(pdf_file)
        print(f"  Extracted {len(text)} chars")

        chunks = chunk_text(
            text,
            doc_slug=doc_config["doc_slug"],
            norm_doc=doc_config["norm_doc"],
            version=doc_config["version"],
        )
        print(f"  Created {len(chunks)} chunks")
        all_chunks.extend(chunks)

    if not all_chunks:
        print("No chunks created. Check norms_dir and NORM_DOCS config.")
        sys.exit(1)

    print(f"\nTotal chunks: {len(all_chunks)}")
    print("Computing embeddings...")

    texts_for_embedding = [PASSAGE_PREFIX + c["text"] for c in all_chunks]
    embeddings = model.encode(
        texts_for_embedding, show_progress_bar=True, normalize_embeddings=True
    )
    embeddings = np.array(embeddings, dtype=np.float32)

    print(f"Embeddings shape: {embeddings.shape}")

    # Build FAISS index (IndexFlatIP with normalized vectors = cosine similarity)
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(embeddings)

    # Save
    index_path = output_dir / "index.faiss"
    metadata_path = output_dir / "metadata.json"

    faiss.write_index(index, str(index_path))
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"\nSaved index to {index_path} ({index.ntotal} vectors)")
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent.parent
    norms_dir = project_root / "data" / "norms"
    output_dir = project_root / "data" / "faiss"

    build_index(norms_dir, output_dir)
