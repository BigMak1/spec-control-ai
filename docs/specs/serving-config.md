# Spec: Serving / Config

## Запуск

### Локальная разработка

```bash
# Backend (Python)
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# Frontend (Node.js)
cd frontend
npm install
npm run dev  # порт 3000

# LangFuse (Docker)
docker compose -f docker-compose.langfuse.yml up -d
```

### Docker Compose (полный стек)

```yaml
# docker-compose.yml
services:
  backend:
    build: ./backend
    ports: ["8000:8000"]
    env_file: .env
    volumes:
      - ./data/faiss:/app/data/faiss:ro
      - ./tmp:/app/tmp
      - ./logs:/app/logs
    depends_on: [langfuse]

  frontend:
    build: ./frontend
    ports: ["3000:3000"]
    environment:
      - API_URL=http://backend:8000
    depends_on: [backend]

  langfuse:
    image: langfuse/langfuse:latest
    ports: ["3001:3000"]
    environment:
      - DATABASE_URL=postgresql://langfuse:langfuse@postgres:5432/langfuse
      - NEXTAUTH_SECRET=${LANGFUSE_SECRET}
      - NEXTAUTH_URL=http://localhost:3001
    depends_on: [postgres]

  postgres:
    image: postgres:16
    environment:
      - POSTGRES_USER=langfuse
      - POSTGRES_PASSWORD=langfuse
      - POSTGRES_DB=langfuse
    volumes:
      - pgdata:/var/lib/postgresql/data

volumes:
  pgdata:
```

## Конфигурация

### Переменные окружения (.env)

```bash
# OpenRouter
OPENROUTER_API_KEY=sk-or-...
OPENROUTER_MODEL=anthropic/claude-sonnet

# LangFuse
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=http://localhost:3001
LANGFUSE_SECRET=random-secret-for-nextauth

# App
MAX_FILE_SIZE_MB=20
MAX_PAGES=50
MAX_AGENT_STEPS=15
CIRCUIT_BREAKER_USD=1.0
CONFIDENCE_THRESHOLD=0.7
LOG_LEVEL=INFO

# Paths
FAISS_INDEX_PATH=data/faiss/index.faiss
METADATA_PATH=data/faiss/metadata.json
TMP_DIR=tmp
LOG_DIR=logs
```

### Структура проекта (планируемая)

```
spec-control-ai/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app
│   │   ├── config.py            # Settings from env
│   │   ├── pipeline.py          # Orchestrator
│   │   ├── parser.py            # Document Parser
│   │   ├── anonymizer.py        # PII Anonymizer
│   │   ├── extractor.py         # Parameter Extractor (workflow + LLM)
│   │   ├── checker.py           # Normative Checker (agent)
│   │   ├── reporter.py          # Report Generator
│   │   ├── deanonymizer.py      # De-anonymizer
│   │   ├── retriever.py         # FAISS search
│   │   ├── llm.py               # OpenRouter client wrapper
│   │   ├── tools.py             # Tool definitions
│   │   ├── schemas.py           # Pydantic models
│   │   └── logging_config.py    # Logging setup
│   ├── scripts/
│   │   └── index_norms.py       # Offline indexation script
│   ├── data/
│   │   ├── norms/               # Source normative PDFs
│   │   └── faiss/               # Generated FAISS index + metadata
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── ...                      # Node.js app
│   └── Dockerfile
├── docker-compose.yml
├── docker-compose.langfuse.yml
├── .env.example
└── docs/
```

## Секреты

| Секрет | Где хранится | Как попадает в приложение |
| ------ | ------------ | ------------------------ |
| OPENROUTER_API_KEY | .env (не в git) | env var |
| LANGFUSE_PUBLIC_KEY | .env | env var |
| LANGFUSE_SECRET_KEY | .env | env var |
| LANGFUSE_SECRET | .env | env var (NextAuth) |

`.env` добавлен в `.gitignore`. Для деплоя используется `.env.example` как шаблон.

## Версии моделей

| Модель | ID в OpenRouter | Назначение |
| ------ | --------------- | ---------- |
| Claude Sonnet | anthropic/claude-sonnet | Все LLM-задачи |
| multilingual-e5-large | intfloat/multilingual-e5-large | Embedding (локально) |

Смена модели — изменение `OPENROUTER_MODEL` в `.env`. Код не меняется.
