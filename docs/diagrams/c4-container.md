# C4 Container: SpecControl AI

## Описание

Показывает внутреннее устройство системы на уровне деплоймент-единиц: frontend, backend API, vector store, observability.
Уровень C4 Level 2 — каждый контейнер является отдельной запускаемой единицей.

## Диаграмма

```mermaid
graph TB
    User["Пользователь"]

    subgraph frontend ["Frontend (Node.js)"]
        WebUI["Web UI<br/>Upload документов<br/>Просмотр отчётов"]
    end

    subgraph backend ["Backend (Python + FastAPI)"]
        API["FastAPI<br/>REST API"]
        Pipeline["Pipeline Orchestrator<br/>Управление шагами"]
        AgentLoop["Agent Loop<br/>Tool-use цикл<br/>(OpenAI SDK)"]
    end

    subgraph storage ["Storage"]
        FAISS["FAISS Index<br/>Векторный поиск"]
        Metadata["Metadata JSON<br/>chunk_id - text, norm, section"]
        Logs["Logs<br/>JSON Lines"]
        Tmp["tmp/<br/>Uploaded files (TTL 1h)"]
    end

    subgraph external ["External Services"]
        OpenRouter["OpenRouter API<br/>Claude Sonnet"]
    end

    subgraph observability ["Observability (Docker)"]
        LangFuseServer["LangFuse Server"]
        Postgres["PostgreSQL<br/>(LangFuse data)"]
    end

    User -->|HTTP| WebUI
    WebUI -->|REST API| API
    API --> Pipeline
    Pipeline --> AgentLoop
    AgentLoop -->|"LLM calls"| OpenRouter
    AgentLoop -->|"similarity search"| FAISS
    AgentLoop -->|"chunk lookup"| Metadata
    Pipeline -->|"traces"| LangFuseServer
    Pipeline -->|"system events"| Logs
    API -->|"save uploaded file"| Tmp
    LangFuseServer --> Postgres

    style AgentLoop fill:#fff3e0,stroke:#e65100
    style FAISS fill:#f3e5f5,stroke:#7b1fa2
    style OpenRouter fill:#e3f2fd,stroke:#1565c0
```

## Контейнеры

| Контейнер | Технология | Назначение |
| --------- | ---------- | ---------- |
| Web UI | Node.js | Загрузка документов, отображение отчётов |
| FastAPI | Python | REST API, pipeline orchestration |
| Agent Loop | Python (OpenAI SDK) | Tool-use агентный цикл для LLM |
| FAISS Index | faiss-cpu (Python) | Векторный поиск по нормативной базе |
| Metadata JSON | Файл .json | Маппинг chunk_id — текст и метаданные норматива |
| LangFuse | Docker (self-hosted) | LLM tracing, cost tracking |
| PostgreSQL | Docker | Хранение данных LangFuse |
