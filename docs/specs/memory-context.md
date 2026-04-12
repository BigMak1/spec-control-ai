# Spec: Memory / Context

## Session State

Каждая обработка документа создаёт объект `SessionState`, хранимый in-memory.

### Schema

```python
@dataclass
class SessionState:
    session_id: str                    # UUID v4
    status: SessionStatus              # enum: parsing | anonymizing | extracting | checking | reporting | done | error
    created_at: datetime
    updated_at: datetime

    # Document data
    filename: str
    file_size_bytes: int
    page_count: int
    raw_text: str                      # Полный текст после парсинга
    sections: list[Section]            # Разбивка по секциям

    # PII
    anonymized_text: str               # Текст после анонимизации
    pii_map: dict[str, str]            # Маппинг токен → реальное значение (ТОЛЬКО RAM)

    # Results
    parameters: list[Parameter]        # Извлечённые параметры
    check_results: list[CheckResult]   # Результаты проверки
    report: str                        # Сгенерированный отчёт

    # Budget tracking
    token_usage: int                   # Суммарный расход токенов
    cost_usd: float                    # Суммарная стоимость ($)
    agent_steps: int                   # Суммарное количество шагов агента
    llm_calls: int                     # Количество LLM вызовов
```

### Lifecycle

1. **Создание:** при получении файла через API
2. **Обновление:** после каждого шага pipeline `status` обновляется
3. **Завершение:** `status = done | error`
4. **Уничтожение:** после отправки ответа пользователю. `pii_map` зануляется первым

### Concurrent access

- PoC: однопоточная обработка, нет необходимости в locks
- Каждый запрос — отдельная сессия, нет shared state между сессиями

## PII Map Policy

| Аспект | Правило |
| ------ | ------- |
| Хранение | Только RAM, никогда на диск |
| Логирование | Запрещено — ни в system logs, ни в LangFuse |
| Время жизни | От создания в PII Anonymizer до отправки ответа |
| Уничтожение | Явное `pii_map.clear()` + `del pii_map` после De-anonymizer |
| Содержимое | `{"[PERSON_1]": "Иванов И.И.", "[ADDR_1]": "ул. Ленина, 5", ...}` |

## Context Budget

### Проблема

Документ до 50 страниц может содержать ~25-50K tokens. Context window Claude Sonnet (200K) вмещает это, но каждый вызов стоит денег. Нужна стратегия управления контекстом.

### Стратегия для Parameter Extractor (workflow)

1. Документ разбит на секции в Document Parser (секции уже в `session.sections`)
2. Детерминированный цикл по секциям — каждая секция обрабатывается отдельным LLM-вызовом
3. Каждый вызов LLM получает: system prompt (~500 tokens) + текст секции (~1-3K tokens)
4. Итого на вызов: ~2-4K tokens input
5. Контекст между вызовами не накапливается (каждый вызов независим) — это дешевле агентного цикла, где conversation history растёт с каждым шагом

### Стратегия для Normative Checker

1. Агент получает один параметр + его контекст (~200 tokens)
2. Tool `search_norms` возвращает top-5 chunks (~2-4K tokens)
3. Если нужен контекст — `get_norm_chunk` добавляет соседние чанки (~1K tokens)
4. Итого на параметр: ~3-6K tokens input per LLM call, до 3 итераций = ~9-18K

### Budget tracking

```python
# Перед каждым LLM вызовом:
if session.cost_usd >= 1.0:
    raise CircuitBreakerError("Budget exceeded")
if session.agent_steps >= 15:
    raise MaxStepsError("Max agent steps reached")  # только Normative Checker

# После каждого LLM вызова:
session.token_usage += response.usage.total_tokens
session.cost_usd += calculate_cost(response.usage, model="anthropic/claude-sonnet")
session.llm_calls += 1
# agent_steps инкрементируется только в Normative Checker (агентный цикл)
```
