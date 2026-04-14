# Normative Checker (Agentic RAG) — Design Spec

TASK_7. Единственный настоящий агент в системе: LLM автономно формулирует поисковые запросы к FAISS, оценивает релевантность чанков, при необходимости переформулирует запрос и выносит вердикт (PASS / FAIL / MANUAL).

---

## 1. Архитектура agent loop

Все параметры документа проверяются в одном agent loop (батч-подход). Агент получает полный список параметров и сам решает порядок проверки. Результаты подаются инкрементально через tool `submit_verdict`.

```
check_norms(session, llm, retriever)
  |
  +-- Собрать system prompt + user prompt (все параметры)
  +-- messages = [system, user]
  |
  +-- while step < max_steps AND cost < budget:
  |     |
  |     +-- response = llm.chat(messages, tools=TOOL_SCHEMAS)
  |     |
  |     +-- if no tool_calls -> break (finish_reason="stop")
  |     |
  |     +-- for each tool_call:
  |     |     +-- execute tool (search_norms / get_norm_chunk / compare_values / submit_verdict)
  |     |     +-- if submit_verdict -> validate chunk_id, apply confidence threshold, save CheckResult
  |     |     +-- append tool result to messages
  |     |
  |     +-- step += 1
  |     +-- session.cost_usd = llm.usage.cost_usd
  |
  +-- Параметры без вердикта -> CheckResult(status=MANUAL)
  +-- return session.check_results
```

### Лимиты

- `max_steps = min(3 * len(parameters), settings.max_agent_steps)` — динамический с потолком 15
- `circuit_breaker_usd` — проверка бюджета **перед** каждым шагом (до `llm.chat`). Шаг не прерывается в середине запроса, поэтому `session.cost_usd` может превысить лимит на стоимость одного шага — это осознанный trade-off (дешевле и проще, чем mid-request cancellation)
- При превышении любого лимита цикл прерывается, непроверенные параметры получают статус MANUAL

---

## 2. Tools (backend/app/tools.py)

4 инструмента: 3 информационных + 1 для вердикта. Tool schemas в формате OpenAI function calling, хранятся в `TOOL_SCHEMAS`.

### search_norms

- **Вход:** `query: str`, `top_k: int = 5`, `filter_doc: str | None`
- **Выход:** список `{chunk_id, norm_doc, section, title, score, text_preview}`
- **Внутри:** `retriever.search_norms()`, text обрезается до 300 символов

### get_norm_chunk

- **Вход:** `chunk_id: str`
- **Выход:** `{chunk: {...}, prev: {...}, next: {...}}`
- **Внутри:** `retriever.get_norm_chunk()`. Если chunk_id не найден — `{"error": "chunk_id not found"}`

### compare_values

- **Вход:** `actual: str`, `required: str`, `comparison_type: str` (gte / lte / eq / contains)
- **Выход:** `{match: bool, actual_parsed, required_parsed, explanation}`
- **Детерминированный.** Парсит числа и сравнивает. При неудаче — `{match: null, explanation: "cannot parse numerically, manual comparison needed"}`

### submit_verdict

- **Вход:** `parameter_name: str`, `status: "PASS" | "FAIL" | "MANUAL"`, `norm_reference: str`, `norm_requirement: str`, `source_chunk_id: str`, `confidence: float`, `explanation: str`
- **Логика:**
  1. Проверить наличие всех 7 required-полей -> если отсутствуют, вернуть `{"error": "missing required fields: <list>"}`, вердикт не сохраняется
  2. Проверить `source_chunk_id` в metadata -> если не существует, вернуть `{"error": "source_chunk_id '...' not found"}`, вердикт не сохраняется
  3. Проверить валидность `status` (PASS/FAIL/MANUAL) и приводимость `confidence` к числу -> ошибка при невалидных значениях
  4. `confidence < 0.7` -> принудительно `status = MANUAL`, в explanation добавляется `"[low confidence -> MANUAL]"`
  5. Найти `Parameter` по `parameter_name` -> собрать `CheckResult` -> сохранить
  6. Вернуть `{"status": "ok", "parameter": <name>, "verdict": <status>}`

---

## 3. Prompt (backend/app/prompts/checker.py)

### System prompt

```
Ты — инженер-нормоконтролёр. Задача: проверить технические параметры
из проектной документации на соответствие российским нормативам
(ПУЭ, СП, ГОСТ).

Доступные инструменты:
- search_norms — поиск по нормативной базе
- get_norm_chunk — получить полный текст пункта норматива с контекстом
- compare_values — детерминированное сравнение значений
- submit_verdict — подать вердикт по параметру

Стратегия работы:
1. Для каждого параметра сформулируй поисковый запрос к нормативной базе
2. Оцени релевантность найденных чанков. Если нерелевантны — переформулируй запрос
3. При необходимости запроси соседние чанки через get_norm_chunk
4. Используй compare_values для числовых сравнений
5. Подай вердикт через submit_verdict

Правила вердиктов:
- PASS — параметр соответствует нормативу
- FAIL — параметр нарушает требование норматива
- MANUAL — не удалось найти релевантный норматив или недостаточно уверенности

Обязательно:
- source_chunk_id — реальный ID чанка из результатов поиска
- confidence — от 0.0 до 1.0, насколько ты уверен в вердикте
- explanation — краткое обоснование на русском языке
- Проверь ВСЕ параметры, не пропускай ни одного
```

### User prompt

Формируется динамически функцией `make_user_prompt(parameters: list[Parameter])`:

```
Проверь следующие параметры из проектной документации:

1. Тип кабеля распределительных сетей = "АВВГнг(А)-LSLTx"
   Контекст: линии систем рабочего освещения детского сада
   Источник: стр. 15

2. Сечение PE-проводника = "1.5" мм²
   Контекст: проводка от коробки КРЗ в санузле
   Источник: стр. 14
```

---

## 4. Anti-hallucination и edge cases

### Верификация chunk_id

При вызове `submit_verdict`:
1. `source_chunk_id` проверяется через `retriever.get_norm_chunk(chunk_id)`
2. Если `None` -> вердикт отклоняется, агенту возвращается ошибка
3. Агент может повторить поиск и подать вердикт с правильным ID

### Confidence threshold

- `confidence < 0.7` -> статус принудительно перезаписывается на `MANUAL`
- В explanation добавляется `"[low confidence -> MANUAL]"`

### Непроверенные параметры

После завершения agent loop:
- Собрать `parameter_name` из всех принятых вердиктов
- Параметры без вердикта -> `CheckResult(status=MANUAL, confidence=0.0, explanation="Агент не вынес вердикт в пределах лимита шагов", source_chunk_id="", norm_reference="", norm_requirement="")`

### Матчинг parameter_name

1. Exact match по `Parameter.name`
2. Fuzzy match — поиск подстроки в обоих направлениях. При множественных совпадениях берётся первый не использованный параметр
3. Не найден -> вердикт отклоняется с ошибкой `{"error": "parameter 'xxx' not found"}`

---

## 5. Тестирование и eval

### Unit-тесты (tests/test_checker.py)

Без реальных LLM-вызовов, мокаем LLMClient:

1. Agent loop завершается по finish_reason="stop"
2. Tool dispatch: search_norms вызывает retriever с правильными аргументами
3. submit_verdict сохраняет CheckResult с валидным chunk_id
4. Невалидный chunk_id отклоняется
5. Confidence < 0.7 -> MANUAL
6. Лимит шагов: бесконечные tool calls -> прерывание, оставшиеся -> MANUAL
7. Budget exceeded -> прерывание
8. compare_values: числовые сравнения (gte, lte, eq), нечисловые -> match=null
9. Fuzzy match parameter_name

### Eval-тест (tests/test_checker_eval.py)

С реальным LLM (`@pytest.mark.skipif` без OPENROUTER_API_KEY):

1. Загрузить ground_truth.json из `data/samples/`
2. Собрать SessionState с expected_parameters
3. Прогнать `check_norms()` с реальным retriever и LLM
4. Сравнить вердикты с expected_violations:
   - **Recall** — доля expected_violations с вердиктом FAIL
   - **Precision** — доля FAIL-вердиктов, совпадающих с expected_violations
5. Вывести метрики в stdout

---

## 6. Файловая структура

```
backend/app/
  checker.py              # check_norms(), agent loop
  tools.py                # TOOL_SCHEMAS, execute_tool(), compare_values, submit_verdict logic
  prompts/
    checker.py            # SYSTEM_PROMPT, make_user_prompt()

tests/
  test_checker.py         # unit-тесты с мокам LLM
  test_checker_eval.py    # eval с реальным LLM
```

---

## 7. Решения, принятые в ходе дизайна

| Вопрос | Решение | Обоснование |
|--------|---------|-------------|
| Обработка параметров | Все в одном agent loop (батч) | Экономия токенов, агент переиспользует найденные чанки |
| Лимит итераций | `min(3 * N, max_agent_steps)` | Масштабируется с количеством параметров, но с потолком |
| Формат вердикта | Tool `submit_verdict` (инкрементальный) | Устойчивость к обрывам, трейсинг каждого вердикта |
| Реализация agent loop | Простой while-цикл с tool dispatch | Минимум абстракций, полный контроль, соответствует задаче "настоящий агент" |
