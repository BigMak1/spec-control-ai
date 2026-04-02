# C4 Component: SpecControl AI Backend

## Описание

Внутреннее устройство Python backend — 6 модулей pipeline и их взаимодействие.
Уровень C4 Level 3. Зелёный цвет — детерминированные модули, оранжевый — агентные.

## Диаграмма

```mermaid
graph TB
    API["FastAPI API Layer<br/>POST /check"]

    subgraph pipeline ["Pipeline (sequential)"]
        direction TB
        Parser["Document Parser<br/>PyMuPDF + python-docx<br/>[DETERMINISTIC]"]
        PII["PII Anonymizer<br/>Regex + Natasha NER<br/>[DETERMINISTIC]"]
        Extractor["Parameter Extractor<br/>Claude Sonnet + tool-use<br/>[AGENT]"]
        Checker["Normative Checker<br/>Agentic RAG<br/>[AGENT]"]
        Reporter["Report Generator<br/>Claude Sonnet single call<br/>[LLM CALL]"]
        DeAnon["De-anonymizer<br/>Token mapping<br/>[DETERMINISTIC]"]

        Parser -->|"raw text + sections"| PII
        PII -->|"anonymized text + pii_map"| Extractor
        Extractor -->|"List[Parameter]"| Checker
        Checker -->|"List[CheckResult]"| Reporter
        Reporter -->|"report with tokens"| DeAnon
    end

    subgraph extractor_tools ["Extractor Tools"]
        ET1["extract_from_chunk()"]
        ET2["list_sections()"]
        ET3["get_chunk()"]
        ET4["validate_parameters()"]
    end

    subgraph checker_tools ["Checker Tools"]
        CT1["search_norms()"]
        CT2["get_norm_chunk()"]
        CT3["compare_values()"]
    end

    API --> Parser
    DeAnon -->|"final report"| API
    Extractor -.->|"tool-use"| extractor_tools
    Checker -.->|"tool-use"| checker_tools
    CT1 -->|"query"| FAISS["FAISS"]
    CT2 -->|"lookup"| Meta["Metadata JSON"]

    style Extractor fill:#fff3e0,stroke:#e65100
    style Checker fill:#fff3e0,stroke:#e65100
    style Parser fill:#e8f5e9,stroke:#2e7d32
    style PII fill:#e8f5e9,stroke:#2e7d32
    style Reporter fill:#e8f5e9,stroke:#2e7d32
    style DeAnon fill:#e8f5e9,stroke:#2e7d32
```

## Модули

| Модуль | Тип | Вход | Выход |
| ------ | --- | ---- | ----- |
| Document Parser | Deterministic | PDF/DOCX file | raw text + sections |
| PII Anonymizer | Deterministic | raw text | anonymized text + pii_map |
| Parameter Extractor | Agent (tool-use) | anonymized text | List[Parameter] |
| Normative Checker | Agent (Agentic RAG) | List[Parameter] | List[CheckResult] |
| Report Generator | LLM call (single) | List[CheckResult] | report text (с токенами PII) |
| De-anonymizer | Deterministic | report + pii_map | final report |
