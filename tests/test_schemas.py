from backend.app.schemas import (
    CheckResult,
    ChunkMetadata,
    ChunkResult,
    Parameter,
    Section,
    SessionState,
)


def test_section_creation():
    s = Section(name="Электроснабжение", text="Текст секции...", page_start=1, page_end=3)
    assert s.name == "Электроснабжение"
    assert s.page_start == 1


def test_parameter_creation():
    p = Parameter(
        name="Сечение кабеля",
        value="2.5",
        unit="мм²",
        context="линия ванной комнаты",
        source_page=5,
        source_text="Кабель ВВГнг 3x2.5 мм²",
    )
    assert p.name == "Сечение кабеля"
    assert p.unit == "мм²"


def test_parameter_optional_unit():
    p = Parameter(
        name="Тип прокладки",
        value="открытая",
        unit=None,
        context="коридор",
        source_page=3,
        source_text="Открытая прокладка кабеля",
    )
    assert p.unit is None


def test_chunk_metadata_creation():
    cm = ChunkMetadata(
        chunk_id="pue_7_1_34_001",
        norm_doc="ПУЭ 7-е изд.",
        section="7.1.34",
        title="Сечения кабелей",
        page=142,
        text="Минимальное сечение жил кабелей...",
        version="2003",
        status="действующий",
    )
    assert cm.chunk_id == "pue_7_1_34_001"
    assert cm.norm_doc == "ПУЭ 7-е изд."


def test_chunk_result_creation():
    cm = ChunkMetadata(
        chunk_id="pue_7_1_34_001",
        norm_doc="ПУЭ 7-е изд.",
        section="7.1.34",
        title="Сечения кабелей",
        page=142,
        text="Текст...",
        version="2003",
        status="действующий",
    )
    cr = ChunkResult(metadata=cm, score=0.85)
    assert cr.score == 0.85


def test_check_result_creation():
    p = Parameter(
        name="Сечение",
        value="2.5",
        unit="мм²",
        context="линия",
        source_page=5,
        source_text="Кабель 2.5",
    )
    cr = CheckResult(
        parameter=p,
        status="PASS",
        norm_reference="ПУЭ 7.1.34",
        norm_requirement="≥ 2.5 мм²",
        source_chunk_id="pue_7_1_34_001",
        confidence=0.85,
        explanation="Соответствует",
    )
    assert cr.status == "PASS"
    assert cr.confidence == 0.85


def test_check_result_invalid_status():
    import pytest

    p = Parameter(
        name="Сечение",
        value="2.5",
        unit="мм²",
        context="линия",
        source_page=5,
        source_text="Кабель 2.5",
    )
    with pytest.raises(ValueError):
        CheckResult(
            parameter=p,
            status="INVALID",
            norm_reference="ПУЭ 7.1.34",
            norm_requirement="≥ 2.5 мм²",
            source_chunk_id="pue_7_1_34_001",
            confidence=0.85,
            explanation="Текст",
        )


def test_session_state_defaults():
    ss = SessionState(session_id="test-123")
    assert ss.status == "parsing"
    assert ss.parameters == []
    assert ss.check_results == []
    assert ss.token_usage == 0
    assert ss.cost_usd == 0.0
    assert ss.agent_steps == 0
