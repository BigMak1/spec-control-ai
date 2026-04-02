# Data Flow: SpecControl AI

## Описание

Показывает путь данных через систему: какие данные на каждом этапе, что хранится, что логируется, что удаляется.
Пунктирные стрелки — запись/чтение в хранилище или трассировка (не основной поток обработки).

## Диаграмма

```mermaid
flowchart LR
    subgraph input ["Input"]
        File["PDF/DOCX<br/>(до 20MB, до 50 стр.)"]
    end

    subgraph processing ["Processing Pipeline"]
        direction TB
        P1["Parser<br/>raw text + sections"]
        P2["PII Anonymizer<br/>clean text + pii_map"]
        P3["Param Extractor<br/>List[Parameter] JSON"]
        P4["Norm Checker<br/>List[CheckResult] JSON"]
        P5["Report Generator<br/>report text"]
        P6["De-anonymizer<br/>final report"]
        P1 --> P2 --> P3 --> P4 --> P5 --> P6
    end

    subgraph stores ["Storage"]
        direction TB
        TMP["tmp/<br/>uploaded file<br/>TTL: 1 hour"]
        RAM["RAM only<br/>pii_map<br/>destroyed after session"]
        FAISS_S["FAISS + Metadata<br/>normative chunks<br/>read-only"]
    end

    subgraph logging ["Logging"]
        direction TB
        SysLog["Python Logs<br/>JSON Lines<br/>system events, errors"]
        LF["LangFuse<br/>LLM traces, costs<br/>agent steps, RAG queries"]
    end

    subgraph output ["Output"]
        Report["Отчёт с PII<br/>(только пользователю)"]
    end

    File -->|"save to tmp/"| TMP
    File --> P1
    P2 -.->|"pii_map in RAM"| RAM
    P4 -.->|"search"| FAISS_S
    P3 -.->|"trace"| LF
    P4 -.->|"trace"| LF
    P5 -.->|"trace"| LF
    P1 -.->|"event"| SysLog
    P6 --> Report
```

## Классификация данных

| Данные | Где хранятся | Срок жизни | Содержит PII |
| ------ | ------------ | ---------- | ------------ |
| Загруженный файл | tmp/ | 1 час / конец сессии | Да — не отправляется в LLM |
| Raw text | RAM (SessionState) | До конца сессии | Да — не отправляется в LLM |
| pii_map | RAM only | До конца сессии | Да — не логируется |
| Anonymized text | RAM (SessionState) | До конца сессии | Нет |
| Parameters JSON | RAM + LangFuse trace | До конца сессии / 30 дней | Нет (анонимизирован) |
| CheckResults JSON | RAM + LangFuse trace | До конца сессии / 30 дней | Нет |
| Report (с токенами) | RAM | До конца сессии | Нет |
| Final report (с PII) | Отдаётся пользователю | Не хранится на сервере | Да — не логируется |
| FAISS index | Файл .faiss | Постоянно (read-only) | Нет |
| Metadata | Файл .json | Постоянно (read-only) | Нет |
| System logs | logs/ (JSON Lines) | 30 дней ротация | Нет |
| LLM traces | LangFuse (PostgreSQL) | Настраиваемо | Нет (анонимизированы) |
