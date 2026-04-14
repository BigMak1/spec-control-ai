# SpecControl AI — System Design

Документ описывает архитектуру PoC-системы автоматизированного нормоконтроля технической документации. Система принимает PDF/DOCX-файл, извлекает технические параметры, сверяет их с нормативной базой через Agentic RAG и формирует отчёт о несоответствиях.

---

## 1. Ключевые архитектурные решения

### Гибридная архитектура: детерминированный pipeline + Agentic RAG

Система построена как последовательный pipeline из 6 шагов. Большинство шагов — детерминированные (парсинг, анонимизация, деанонимизация) или детерминированные с LLM-вызовами внутри (извлечение параметров, генерация отчёта). Единственный настоящий агент — **Normative Checker**, который автономно формулирует поисковые запросы, оценивает релевантность результатов и принимает решение о переформулировке.

**Принцип:** агент (LLM в цикле, автономно выбирающий инструменты) используется только там, где нельзя описать детерминированный алгоритм. Извлечение параметров — предсказуемый цикл по секциям документа с LLM-вызовом на каждом шаге, поэтому реализовано как workflow, а не агент. Это снижает стоимость, сложность и риск зацикливания.

### Таблица ключевых решений

| Решение | Выбор | Обоснование |
|---------|-------|-------------|
| Агентный фреймворк | Собственный agent loop на OpenAI Python SDK | Полный контроль над промптами, tool-use циклом и бюджетом. Минимум внешних зависимостей |
| LLM | Claude Sonnet через OpenRouter API | Гибкость смены модели без изменения кода. Бюджет < $0.50/документ |
| Vector store | FAISS + JSON metadata | Простота, скорость, достаточно для 3-5 нормативных документов PoC-масштаба |
| Embedding | multilingual-e5-large (локально, CPU) | Нет зависимости от второго API-провайдера. CPU достаточно для PoC |
| Backend | Python + FastAPI | Все ключевые библиотеки (PyMuPDF, FAISS, sentence-transformers) нативно на Python |
| Tooling | uv (package manager) + ruff (linter/formatter) | Быстрая установка зависимостей, единый lock-файл, линтинг и форматирование в одном инструменте |
| Frontend | Node.js | Гибкость построения UI |
| Observability | LangFuse self-hosted (Docker) | LLM tracing, cost tracking без лимитов. Данные остаются локально |
| System logging | Python logging → JSON Lines | Системные события и ошибки (без PII) |

**Почему не LangGraph/CrewAI:** готовые агентные фреймворки добавляют абстракции, которые затрудняют контроль бюджета и отладку. При фиксированном небольшом pipeline это overhead без пользы.

**Почему FAISS, а не ChromaDB:** FAISS — файловый индекс, не требует отдельного сервиса. При 3-5 нормативных документах это достаточно. ChromaDB разумно рассмотреть при росте базы.

**Почему OpenRouter вместо прямого Anthropic API:** единая точка интеграции позволяет менять модель (Claude → GPT-4o → Mistral) одним параметром без изменения кода. Ценообразование прозрачно.

---

## 2. Модули и их роли

Система состоит из 6 модулей, выполняемых последовательно оркестратором.

### 2.1. Document Parser (детерминированный)

| Поле | Значение |
|------|----------|
| Вход | PDF/DOCX файл (до 50 страниц, до 20 MB) |
| Выход | `raw_text: str`, `sections: List[Section]` |
| Тип | Детерминированный |
| Технология | PyMuPDF (PDF), python-docx (DOCX) |
| Ограничения | Только документы с текстовым слоем. Сканы без OCR-слоя — отказ с сообщением пользователю |

Модуль извлекает текст и структурно разбивает документ на секции (по заголовкам). Если текст не извлечён (`len(text) < threshold`) — pipeline останавливается с понятным сообщением об ошибке.

### 2.2. PII Anonymizer (детерминированный)

| Поле | Значение |
|------|----------|
| Вход | `raw_text: str` |
| Выход | `anonymized_text: str`, `pii_map: dict` (in-memory) |
| Тип | Детерминированный |
| Технология | regex + Natasha (NER для русского языка) |
| Ограничения | `pii_map` хранится только в RAM, не логируется, уничтожается после сессии |

Замены: ФИО → `[PERSON]`, адреса → `[ADDR]`, телефоны → `[TEL]`, email → `[EMAIL]`, ИНН → `[INN]`. Анонимизация обязательна: в LLM API отправляется только обезличенный текст.

### 2.3. Parameter Extractor (детерминированный workflow + LLM)

| Поле | Значение |
|------|----------|
| Вход | `anonymized_text: str`, `sections: List[Section]` |
| Выход | `parameters: List[Parameter]` (JSON, schema-validated) |
| Тип | Детерминированный цикл по секциям с LLM-вызовом на каждом шаге |
| Технология | OpenAI Python SDK (structured output, без tool-use loop) |
| Ограничения | Max 50K токенов суммарно, retry при невалидном JSON — до 2 раз на секцию |

**Почему не агент:** для документа ≤50 страниц алгоритм обхода секций полностью предсказуем — перебрать все секции последовательно и извлечь параметры из каждой. LLM не нужно принимать решение «какую секцию читать следующей» — это решение тривиально и запрограммировано в коде. LLM используется только для семантического извлечения параметров из текста (structured output), что является одиночным вызовом, а не агентным циклом.

**Алгоритм:**
1. Получить список секций из Document Parser
2. Для каждой секции: отправить текст секции в LLM → получить `List[Parameter]` (structured output)
3. Объединить результаты, провалидировать по JSON-схеме
4. При невалидном JSON — retry до 2 раз с уточнённым промптом

### 2.4. Normative Checker (Agentic RAG)

| Поле | Значение |
|------|----------|
| Вход | `parameters: List[Parameter]` |
| Выход | `check_results: List[CheckResult]` (JSON, schema-validated) |
| Тип | Агентный RAG-цикл на каждый параметр |
| Технология | FAISS + e5-large + Claude Sonnet |
| Ограничения | Max 3 поисковых итерации на параметр, confidence < 0.7 → `MANUAL`, обязательное поле `source_chunk_id` |

Ключевой модуль с точки зрения качества. Агент адаптивно формулирует запрос, оценивает релевантность найденных чанков и при необходимости переформулирует запрос — до 3 попыток. Результат без верифицированного `source_chunk_id` не принимается (защита от галлюцинаций).

**Инструменты агента:**
- `search_norms(query, top_k, filter_doc?)` — поиск по FAISS с опциональной фильтрацией по документу
- `get_norm_chunk(chunk_id)` — получить полный чанк с соседними для контекста
- `compare_values(actual, required)` — вспомогательное структурированное сравнение значений

### 2.5. Report Generator (одиночный вызов LLM)

| Поле | Значение |
|------|----------|
| Вход | `check_results: List[CheckResult]` |
| Выход | `report_text: str` (с PII-токенами вместо реальных данных) |
| Тип | Один вызов Claude Sonnet по шаблону — не агент |
| Технология | OpenAI Python SDK (single completion) |
| Ограничения | Без итераций, детерминированный шаблон |

Для каждого несоответствия шаблон формирует: параметр → значение в документе → требование норматива → пункт норматива → рекомендация. Формат предсказуем, поэтому агентная логика не нужна.

### 2.6. De-anonymizer (детерминированный)

| Поле | Значение |
|------|----------|
| Вход | `report_text: str` (с токенами) + `pii_map: dict` |
| Выход | Финальный отчёт с реальными данными |
| Тип | Детерминированный |
| Технология | Строковая замена токенов по `pii_map` |
| Ограничения | Выполняется в самом конце pipeline, после всех LLM-вызовов |

Токены `[PERSON]`, `[ADDR]` и др. заменяются на исходные значения из `pii_map`. Реальные PII-данные ни разу не покидают границ системы до этого шага.

---

## 3. Основной workflow

```
Upload (PDF/DOCX)
  │
  ▼
[1] Document Parser ──────────── DETERMINISTIC
  │   file → raw_text + sections
  │
  ├── FAIL: нет текстового слоя → STOP
  │         Сообщение: "Документ не содержит текстового слоя"
  ▼
[2] PII Anonymizer ──────────── DETERMINISTIC
  │   raw_text → anonymized_text + pii_map (RAM only)
  │
  ├── FAIL: NER сервис недоступен → STOP (не отправлять неанонимизированный текст в API)
  ▼
[3] Parameter Extractor ───────── WORKFLOW + LLM (loop over sections)
  │   anonymized_text + sections → List[Parameter]
  │
  ├── FAIL: LLM вернул невалидный JSON для секции → retry 2x → пропустить секцию
  ├── FAIL: API недоступен → retry 3x exp.backoff → STOP
  ▼
[4] Normative Checker ─────────── AGENTIC RAG (per parameter, max 3 iter each)
  │   List[Parameter] → List[CheckResult]
  │
  ├── FAIL: норматив не найден → parameter.status = MANUAL
  ├── FAIL: невалидный chunk_id (галлюцинация) → отклонить, retry поиск
  ├── FAIL: cost > $1 → circuit breaker → partial result
  ▼
[5] Report Generator ──────────── LLM CALL (single prompt)
  │   List[CheckResult] → report_text (с токенами вместо PII)
  │
  ├── FAIL: API недоступен → retry 3x exp.backoff (2/4/8 с) → STOP
  ▼
[6] De-anonymizer ─────────────── DETERMINISTIC
  │   tokens → реальные PII
  ▼
Response to User
```

---

## 4. State / Memory / Context Handling

### SessionState

Каждый запрос на обработку документа создаёт объект сессии, который передаётся между модулями:

```python
@dataclass
class SessionState:
    session_id: str          # UUID, уникальный идентификатор сессии
    status: str              # parsing | extracting | checking | reporting | done | error
    raw_text: str            # Исходный текст документа (не логируется)
    anonymized_text: str     # Текст после анонимизации
    pii_map: dict            # Только в RAM, не сериализуется, не логируется
    sections: list           # Список секций документа
    parameters: list         # List[Parameter] после Parameter Extractor
    check_results: list      # List[CheckResult] после Normative Checker
    report: str              # Финальный отчёт с реальными данными
    token_usage: int         # Суммарный расход токенов по сессии
    cost_usd: float          # Суммарная стоимость LLM-вызовов в долларах
    agent_steps: int         # Суммарное количество шагов агентов
```

Сессия хранится in-memory во время обработки. После завершения — `pii_map` уничтожается немедленно; остальные поля могут быть сохранены в SQLite для трассировки (кроме `raw_text` и `pii_map`).

### Хранилища

| Хранилище | Тип | Назначение |
|-----------|-----|------------|
| FAISS index (`.faiss`) | Файл, read-only в runtime | Векторный индекс нормативной базы |
| Metadata store (`.json`) | Файл, read-only в runtime | `chunk_id` → text, norm_doc, section, page, version, status |
| Logs (`logs/`) | JSON Lines, append-only | Системные события, LLM calls (без PII и полного текста) |
| `tmp/` | Ephemeral | Загруженные файлы, TTL 1 час |
| LangFuse (Docker) | PostgreSQL | LLM traces, cost tracking, evaluation metrics |

### Управление контекстом в агентах (Context Budget Policy)

LLM-вызовы не получают весь документ сразу — текст подаётся по секциям.

**Parameter Extractor (workflow):** детерминированный цикл по секциям. Каждый LLM-вызов получает текст одной секции (~1-3K токенов). Бюджет проверяется перед каждым вызовом.

**Normative Checker (агент):** агентный цикл с tool-use. Агент сам выбирает инструменты и стратегию поиска. Бюджет проверяется перед каждым шагом агента.

```python
def check_budget(state: SessionState) -> bool:
    if state.cost_usd >= 1.0:          # Circuit breaker: $1 на запрос
        raise BudgetExceededException()
    if state.agent_steps >= 15:        # Circuit breaker: 15 шагов (только Normative Checker)
        raise StepLimitExceededException()
    return True
```

Circuit breaker по шагам (`agent_steps`) считает только шаги агента Normative Checker, т.к. Parameter Extractor — детерминированный workflow и не может зациклиться.

---

## 5. Retrieval-контур

### Offline: индексация нормативной базы (однократно)

Индексация выполняется заранее и не входит в runtime pipeline. Порядок:

1. Загрузить нормативные PDF (ПУЭ, ГОСТ Р 50571, СП — 3–5 документов)
2. Разбить на чанки по пунктам/разделам норматива:
   - Размер чанка: ~500–800 токенов
   - Перекрытие: ~100 токенов
   - Заголовок пункта включается в каждый чанк для контекста при поиске
3. Векторизация: `multilingual-e5-large` (CPU, несколько минут однократно)
4. Сохранить: FAISS index (`.faiss`) + metadata (`.json`)

### Схема метаданных чанка

```json
{
  "chunk_id": "pue_7_1_34_002",
  "norm_doc": "ПУЭ 7-е изд.",
  "section": "7.1.34",
  "title": "Сечения кабелей",
  "page": 142,
  "text": "Минимальное сечение жил кабелей...",
  "version": "2003",
  "status": "действующий"
}
```

Поля `norm_doc` и `section` позволяют агенту фильтровать поиск (`filter_doc` в `search_norms`) и верифицировать ссылку на норматив через `source_chunk_id`.

### Runtime: Agentic RAG поиск

Для каждого параметра из `List[Parameter]`:

1. Агент формирует поисковый запрос на основе параметра и его контекста
2. Embedding запроса через e5-large (CPU, ~200 мс)
3. FAISS similarity search (top_k=5, ~1 мс)
4. Metadata lookup: `chunk_id` → полная информация о чанке
5. Агент оценивает релевантность найденных чанков:
   - **Релевантно** → использовать для сравнения с параметром
   - **Нерелевантно** → переформулировать запрос (до 3 попыток)
   - **Частично релевантно** → запросить соседние чанки через `get_norm_chunk(chunk_id)`
6. Агент выносит вердикт: `PASS` / `FAIL` / `MANUAL` с указанием `source_chunk_id`

Если за 3 итерации релевантный чанк не найден → `status = MANUAL`, пользователь получает пометку «норматив не найден в базе».

---

## 6. Tool/API-интеграции

### OpenRouter API (LLM)

Используется через OpenAI Python SDK с заменой base URL:

```python
from openai import OpenAI

client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
)

# Вызов Claude Sonnet
response = client.chat.completions.create(
    model="anthropic/claude-sonnet-4-5",
    messages=[...],
    tools=[...],
)
```

### Parameter Extractor — внутренние функции (не agent tools)

Parameter Extractor реализован как детерминированный workflow. Перечисленные функции вызываются кодом pipeline, а не LLM через tool-use:

| Функция | Сигнатура | Описание |
|---------|-----------|----------|
| `get_sections` | `(session) → List[Section]` | Получить секции из Document Parser (уже готовы в SessionState) |
| `extract_parameters_from_section` | `(section_text: str) → List[Parameter]` | LLM-вызов: structured output извлечения параметров из текста секции |
| `validate_parameters` | `(params: List[Parameter]) → ValidationResult` | JSON Schema валидация итогового списка параметров |

### Инструменты Normative Checker (agent tools)

| Tool | Сигнатура | Описание |
|------|-----------|----------|
| `search_norms` | `(query: str, top_k: int, filter_doc?: str) → List[ChunkResult]` | FAISS поиск с опциональной фильтрацией по документу |
| `get_norm_chunk` | `(chunk_id: str) → {chunk, neighbors}` | Чанк с соседними для полного контекста |
| `compare_values` | `(actual: str, required: str) → CompareResult` | Структурированное сравнение числовых/текстовых значений |

### Формат tool-use (OpenAI SDK)

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_norms",
            "description": "Поиск релевантных пунктов нормативов по запросу",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Поисковый запрос"},
                    "top_k": {"type": "integer", "default": 5},
                    "filter_doc": {"type": "string", "description": "Фильтр по документу (опционально)"}
                },
                "required": ["query"]
            }
        }
    }
]
```

### Контракты данных

**Parameter:**
```json
{
  "name": "Сечение кабеля",
  "value": "2.5",
  "unit": "мм²",
  "context": "линия ванной комнаты",
  "source_page": 5,
  "source_text": "Кабель ВВГнг 3x2.5 мм²..."
}
```

**CheckResult:**
```json
{
  "parameter": { "...": "Parameter object" },
  "status": "PASS | FAIL | MANUAL",
  "norm_reference": "ПУЭ 7.1.34",
  "norm_requirement": "≥ 2.5 мм²",
  "source_chunk_id": "pue_7_1_34_002",
  "confidence": 0.85,
  "explanation": "Сечение кабеля соответствует минимальным требованиям ПУЭ..."
}
```

---

## 7. Failure modes, fallback и guardrails

### Таблица сценариев отказа

| Failure | Где возникает | Как детектируется | Fallback | Что видит пользователь |
|---------|--------------|-------------------|----------|------------------------|
| PDF без текстового слоя | Document Parser | `len(text) < threshold` | Остановка pipeline | «Документ не содержит текстового слоя. Загрузите текстовый PDF.» |
| LLM вернул невалидный JSON | Parameter Extractor / Normative Checker | JSON Schema validation | Retry до 2 раз с уточнённым промптом | После 2 retry — ошибка с кодом |
| Агент превысил лимит итераций | Normative Checker | `iterations > max` (3 на параметр) | Принудительный stop, `status = MANUAL` | «Норматив не найден — требуется ручная проверка» |
| Норматив не найден в базе | Normative Checker (RAG) | `search_score < threshold` после 3 попыток | `status = MANUAL` для параметра | «Норматив не найден — требуется ручная проверка» |
| Галлюцинация (несуществующий `chunk_id`) | Normative Checker | Проверка `chunk_id` в metadata | Отклонить вердикт, retry поиска | Пользователь не видит невалидных ссылок на нормативы |
| LLM API недоступен | Любой LLM-шаг | HTTP 5xx / timeout | Retry 3x с exp. backoff (2/4/8 с) | «Сервис временно недоступен. Попробуйте позже.» |
| Превышен бюджет на запрос | Оркестратор | `cost_usd > $1.0` (circuit breaker) | Partial result, остановка | «Обработка остановлена: документ слишком сложный для автоматической проверки» |
| Prompt injection в документе | PII Anonymizer + валидация output | Regex + аномальный tool call | Санитизация + reject невалидного output | Прозрачно (injection блокируется до LLM) |

### Guardrails — сводка лимитов

| Scope | Параметр | Значение |
|-------|----------|----------|
| Parameter Extractor | Max tokens (суммарно по всем секциям) | 50K |
| Parameter Extractor | Retry on invalid JSON per section | 2× |
| Normative Checker | Max search iterations per parameter | 3 |
| Normative Checker | Confidence threshold (MANUAL) | 0.7 |
| Normative Checker | Max cost per parameter | $0.05 |
| Pipeline (global) | Max document size | 50 страниц / 20 MB |
| Pipeline (global) | Max total cost per request | $1.00 (circuit breaker) |
| Pipeline (global) | Max agent steps total | 15 |
| Pipeline (global) | API retry | 3× exp. backoff (2/4/8 с) |

---

## 8. Технические и операционные ограничения

### Продуктовые метрики (целевые значения PoC)

| Метрика | Целевое значение |
|---------|-----------------|
| Доля найденных реальных несоответствий (recall) | ≥ 70% |
| Доля корректных замечаний из всех выданных (precision) | ≥ 80% |
| Доля успешных запусков без ошибок (reliability) | ≥ 90% |

### Агентные метрики

| Метрика | Целевое значение |
|---------|-----------------|
| Среднее количество шагов агента на документ | ≤ 10 |
| Доля зацикливаний (loops) | < 5% |
| Частота срабатывания fallback-логики | < 15% |

### Технические метрики

| Метрика | Целевое значение |
|---------|-----------------|
| p95 latency полного pipeline | < 180 с |
| Стоимость обработки одного документа (API calls) | < $0.50 |
| Uptime сервиса (в рамках PoC) | > 95% |

### Операционные ограничения

| Параметр | Значение | Обоснование |
|----------|----------|-------------|
| Бюджет на LLM API (PoC) | < $50/мес | Ограничение на стадии прототипа |
| Количество нормативов в базе | 3–5 документов | PoC-масштаб, достаточен для демо |
| Поддерживаемые форматы | PDF, DOCX | Минимальный набор для демонстрации |
| Языки документов | Только русский | Основной язык технической документации; NER-модель Natasha оптимизирована под русский |
| Тип документов | Только текстовые (не сканы, не чертежи) | OCR выходит за рамки PoC |

### Observability

Для мониторинга качества и стоимости используется **LangFuse (self-hosted, Docker Compose)**:

- **LLM tracing:** каждый вызов Claude — prompt, response, latency, tokens, cost
- **Agent tracing:** полный trace агентного цикла — шаги, tool calls, решения
- **RAG tracing:** query → retrieved chunks → scores → оценка релевантности агентом
- **Cost tracking:** автоматический подсчёт через OpenRouter pricing
- **Evaluation:** привязка метрик качества (confidence, PASS/FAIL/MANUAL ratio) к traces

**Системное логирование** (Python logging → JSON Lines в `logs/`):
- Системные события: upload, parser status, количество обнаруженных PII
- Ошибки: stack traces с контекстом (без PII)
- Превышение лимитов: тип, текущее значение, порог

**Что НЕ логируется:** полный текст документа, персональные данные (ФИО, адреса, телефоны, email), содержимое до анонимизации, API-ключи.
