# Spec: Observability / Evals

## LangFuse Tracing

### Интеграция

```python
from langfuse import Langfuse
from langfuse.openai import openai  # drop-in replacement

langfuse = Langfuse(
    public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
    secret_key=os.environ["LANGFUSE_SECRET_KEY"],
    host=os.environ["LANGFUSE_HOST"],
)

# OpenAI SDK с LangFuse обёрткой — автоматический tracing
client = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)
```

### Trace structure

```
Trace: process_document (session_id)
├── Span: parse_document
│   └── Event: text_extracted (pages, chars)
├── Span: anonymize_pii
│   └── Event: pii_detected (count per type)
├── Span: extract_parameters (workflow)
│   ├── Span: section_1
│   │   └── Generation: LLM call (prompt, response, tokens, cost)
│   ├── Span: section_2
│   │   └── Generation: LLM call (prompt, response, tokens, cost)
│   ├── ...
│   └── Event: parameters_extracted (count, sections_processed, sections_skipped)
├── Span: check_norms
│   └── Span: check_parameter (per param)
│       ├── Event: search_norms (query, top_k scores)
│       ├── Generation: LLM call (prompt, response, tokens, cost)
│       └── Event: verdict (status, confidence, chunk_id)
├── Span: generate_report
│   └── Generation: LLM call (prompt, response, tokens, cost)
└── Event: session_complete (total_cost, total_tokens, total_steps)
```

### Scores (привязанные к traces)

| Score name | Type | Описание |
| ---------- | ---- | -------- |
| parameters_count | NUMERIC | Количество извлечённых параметров |
| pass_count | NUMERIC | Количество PASS вердиктов |
| fail_count | NUMERIC | Количество FAIL вердиктов |
| manual_count | NUMERIC | Количество MANUAL вердиктов |
| avg_confidence | NUMERIC | Средний confidence по всем вердиктам |
| total_cost_usd | NUMERIC | Полная стоимость обработки |
| total_agent_steps | NUMERIC | Количество шагов агента |
| processing_time_sec | NUMERIC | Полное время обработки |

## Python Logging

### Конфигурация

```python
import logging
import json

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "module": record.module,
            "message": record.getMessage(),
            "session_id": getattr(record, "session_id", None),
        }
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_data, ensure_ascii=False)
```

### Уровни

| Level | Что логируется | Пример |
| ----- | -------------- | ------ |
| INFO | Ключевые события pipeline | `{"message": "document_parsed", "pages": 15, "chars": 23000}` |
| DEBUG | LLM prompts (anonymized), RAG queries | `{"message": "llm_call", "model": "...", "tokens": 1500}` |
| WARNING | Превышение порогов | `{"message": "budget_warning", "cost": 0.85, "threshold": 1.0}` |
| ERROR | Ошибки с stack traces | `{"message": "llm_invalid_json", "attempt": 2}` |

### Ротация

- Файл: `logs/speccontrol.jsonl`
- Ротация: ежедневно (`TimedRotatingFileHandler`)
- Хранение: 30 дней
- Не логируется: PII, полный текст документа, API ключи

## Что НЕ логируется

- Полный текст загруженного документа
- Персональные данные (ФИО, адреса, телефоны, email)
- Содержимое документа до анонимизации
- API-ключи и секреты
- Содержимое pii_map

## Evaluation (PoC)

### Eval-набор

Для оценки качества системы нужен размеченный eval-набор:

| Компонент | Что размечается | Размер для PoC |
| --------- | --------------- | -------------- |
| Parameter Extractor | Документ + ожидаемые параметры | 5-10 документов |
| Normative Checker | Параметр + ожидаемый вердикт (PASS/FAIL) + пункт норматива | 20-30 пар |

### Метрики качества

| Метрика | Формула | Целевое значение |
| ------- | ------- | ---------------- |
| Recall (параметры) | найденные_верные / все_верные | >= 70% |
| Precision (вердикты) | верные_вердикты / все_вердикты | >= 80% |
| Hallucination rate | невалидные_chunk_id / все_chunk_id | < 5% |
| MANUAL rate | manual_вердикты / все_вердикты | < 30% |

### Запуск eval

```bash
python -m scripts.run_eval --eval-set data/eval/ --output results/eval_YYYY-MM-DD.json
```

Результаты eval привязываются к LangFuse traces через scores для визуализации трендов.
