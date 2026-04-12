from backend.scripts.index_norms import chunk_text, detect_sections


def test_detect_sections_simple():
    text = (
        "7.1.34 Минимальное сечение жил кабелей должно быть не менее 2.5 мм².\n"
        "7.1.35 Для питания группы розеток следует применять кабель с сечением 2.5 мм².\n"
        "7.1.36 Линии освещения выполняются кабелем 1.5 мм².\n"
    )
    sections = detect_sections(text)
    assert len(sections) == 3
    assert sections[0]["section"] == "7.1.34"
    assert "2.5 мм²" in sections[0]["text"]
    assert sections[1]["section"] == "7.1.35"
    assert sections[2]["section"] == "7.1.36"


def test_detect_sections_nested():
    text = "6.2 Общие требования\n6.2.1 Текст пункта один.\n6.2.2 Текст пункта два.\n"
    sections = detect_sections(text)
    assert len(sections) >= 2
    ids = [s["section"] for s in sections]
    assert "6.2.1" in ids
    assert "6.2.2" in ids


def test_detect_sections_no_structure():
    text = "Просто текст без номеров пунктов. Ещё одно предложение. И ещё."
    sections = detect_sections(text)
    assert sections == []


def test_chunk_text_short_sections():
    """Секции короче 800 токенов не дробятся."""
    text = "7.1.34 Короткий пункт про сечение кабелей.\n7.1.35 Ещё один короткий пункт.\n"
    chunks = chunk_text(text, doc_slug="pue", norm_doc="ПУЭ 7-е изд.", version="2003")
    assert len(chunks) == 2
    assert chunks[0]["chunk_id"] == "pue_7_1_34_001"
    assert chunks[0]["norm_doc"] == "ПУЭ 7-е изд."
    assert chunks[1]["chunk_id"] == "pue_7_1_35_001"


def test_chunk_text_long_section_splits():
    """Секция длиннее 800 токенов дробится с overlap."""
    long_body = "Слово " * 1000  # ~1000 токенов
    text = f"7.1.34 {long_body}\n7.1.35 Короткий.\n"
    chunks = chunk_text(text, doc_slug="pue", norm_doc="ПУЭ 7-е изд.", version="2003")
    pue_34_chunks = [c for c in chunks if c["section"] == "7.1.34"]
    assert len(pue_34_chunks) > 1
    assert pue_34_chunks[0]["chunk_id"] == "pue_7_1_34_001"
    assert pue_34_chunks[1]["chunk_id"] == "pue_7_1_34_002"


def test_chunk_text_fallback_no_structure():
    """Текст без структуры дробится fixed-size."""
    text = "Слово " * 500
    chunks = chunk_text(text, doc_slug="doc", norm_doc="Документ", version="2020")
    assert len(chunks) >= 1
    assert chunks[0]["section"] == ""
    assert chunks[0]["chunk_id"].startswith("doc_")


def test_chunk_text_includes_title_in_subchunks():
    """При дроблении длинной секции заголовок включается в каждый подчанк."""
    long_body = "Слово " * 1000
    text = f"7.1.34 Сечения кабелей. {long_body}\n"
    chunks = chunk_text(text, doc_slug="pue", norm_doc="ПУЭ 7-е изд.", version="2003")
    for chunk in chunks:
        assert "7.1.34" in chunk["text"]
