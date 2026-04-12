# Spec: Tools / APIs

## OpenRouter API

### Конфигурация

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)
```

### Модель

| Параметр | Значение |
| -------- | -------- |
| Model ID | anthropic/claude-sonnet (актуальная версия через OpenRouter) |
| Max context | 200K tokens |
| Temperature | 0.0 (детерминизм для reproducibility) |
| Tool-use | Поддерживается через OpenRouter |

### Rate limits и retry

| Параметр | Значение |
| -------- | -------- |
| Retry policy | 3 попытки, exponential backoff (2s, 4s, 8s) |
| Timeout per request | 60 секунд |
| HTTP 429 (rate limit) | Retry с увеличенным backoff |
| HTTP 5xx | Retry |
| HTTP 4xx (кроме 429) | Не retry, ошибка |

## Parameter Extractor — внутренние функции pipeline

Parameter Extractor — детерминированный workflow, не агент. Перечисленные функции вызываются кодом pipeline, а не LLM через tool-use. LLM используется только для structured output (извлечение параметров из текста секции).

### extract_parameters_from_section (LLM-вызов)

- **Вход:** текст секции документа (из `session.sections`)
- **Выход:** `List[Parameter]` (JSON, structured output)
- **LLM:** single call, `response_format={"type": "json_object"}`, temperature=0.0
- **Retry:** до 2 раз при невалидном JSON
- **Ошибки:** невалидный JSON после 2 retry → пропуск секции с warning

### validate_parameters (детерминированная)

- **Вход:** `List[Parameter]` (все извлечённые параметры)
- **Выход:** `ValidationResult` — `{"valid": true/false, "issues": ["..."]}`
- **Реализация:** JSON Schema validation, без LLM

## Tools: Normative Checker (agent tools)

### search_norms

```json
{
  "name": "search_norms",
  "description": "Поиск релевантных нормативных требований по запросу",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {"type": "string", "description": "Поисковый запрос на русском языке"},
      "top_k": {"type": "integer", "default": 5, "description": "Количество результатов"},
      "filter_doc": {"type": "string", "description": "Фильтр по конкретному нормативному документу"}
    },
    "required": ["query"]
  }
}
```

- **Side effects:** нет
- **Timeout:** ~300ms (embedding + FAISS search)
- **Ошибки:** пустой query → error; нет результатов → пустой массив

### get_norm_chunk

```json
{
  "name": "get_norm_chunk",
  "description": "Получить полный текст нормативного фрагмента по его ID, включая контекст (соседние фрагменты)",
  "parameters": {
    "type": "object",
    "properties": {
      "chunk_id": {"type": "string"}
    },
    "required": ["chunk_id"]
  }
}
```

- **Side effects:** нет
- **Ошибки:** несуществующий chunk_id → error (используется для верификации галлюцинаций)

### compare_values

```json
{
  "name": "compare_values",
  "description": "Сравнить фактическое значение параметра с нормативным требованием",
  "parameters": {
    "type": "object",
    "properties": {
      "actual_value": {"type": "string"},
      "actual_unit": {"type": "string"},
      "required_value": {"type": "string"},
      "required_unit": {"type": "string"},
      "comparison_type": {"type": "string", "enum": ["gte", "lte", "eq", "range"]}
    },
    "required": ["actual_value", "required_value", "comparison_type"]
  }
}
```

- **Side effects:** нет
- **Ошибки:** несовместимые единицы → error с описанием

## Prompt Injection защита

| Уровень | Механизм | Реализация |
| ------- | -------- | ---------- |
| L1 | Изоляция | Содержимое документа в `<document>` теге, system prompt отдельно |
| L2 | Санитизация | Regex-удаление паттернов `ignore previous`, `system:`, `<\|im_start\|>` |
| L3 | Валидация output | JSON schema validation на каждый ответ LLM |
| L4 | Мониторинг | Логирование аномальных tool calls (unexpected tool names) |
