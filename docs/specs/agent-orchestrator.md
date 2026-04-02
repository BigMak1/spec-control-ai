# Spec: Agent / Orchestrator

## Pipeline Orchestrator

Orchestrator — детерминированная последовательность шагов. Не агент.

### Последовательность

```python
async def process_document(file: UploadFile) -> Report:
    session = SessionState(file)

    # Step 1: Parse (deterministic)
    session.raw_text, session.sections = parse_document(file)

    # Step 2: Anonymize (deterministic)
    session.anonymized_text, session.pii_map = anonymize_pii(session.raw_text)

    # Step 3: Extract parameters (AGENT)
    session.parameters = await extract_parameters(session)

    # Step 4: Check norms (AGENT per parameter)
    session.check_results = await check_norms(session)

    # Step 5: Generate report (single LLM call)
    session.report = await generate_report(session.check_results)

    # Step 6: De-anonymize (deterministic)
    final_report = deanonymize(session.report, session.pii_map)

    # Cleanup
    session.pii_map.clear()
    return final_report
```

### Правила переходов

| Из | В | Условие | При ошибке |
| -- | -- | ------- | ---------- |
| parsing | anonymizing | Текст извлечён, `len > threshold` | STOP: "нет текстового слоя" |
| anonymizing | extracting | Анонимизация завершена | Невозможно (deterministic) |
| extracting | checking | Parameters получены (full или partial) | partial → продолжаем с warning |
| checking | reporting | CheckResults получены (full или partial) | partial → продолжаем с warning |
| reporting | done | Отчёт сгенерирован | Retry 2x → STOP |

## Agent Loop: Parameter Extractor

### Реализация agent loop

```python
async def extract_parameters(session: SessionState) -> list[Parameter]:
    messages = [{"role": "system", "content": EXTRACTOR_SYSTEM_PROMPT}]
    tools = [extract_from_chunk, list_sections, get_chunk, validate_parameters]
    all_parameters = []

    for iteration in range(MAX_ITERATIONS):  # MAX_ITERATIONS = 10
        check_budget(session)  # raises CircuitBreakerError

        response = await client.chat.completions.create(
            model="anthropic/claude-sonnet",
            messages=messages,
            tools=tool_schemas(tools),
            temperature=0.0,
        )
        session.agent_steps += 1
        track_usage(session, response)

        if response.choices[0].finish_reason == "stop":
            break  # Агент решил, что закончил

        # Process tool calls
        for tool_call in response.choices[0].message.tool_calls:
            result = execute_tool(tool_call, session)
            messages.append(tool_call_result(tool_call, result))

        messages.append(response.choices[0].message)

    # Validate output
    return validate_and_parse_parameters(messages[-1].content)
```

### Stop conditions

| Условие | Действие |
| ------- | -------- |
| Агент вернул `finish_reason == "stop"` | Нормальное завершение |
| `iteration >= 10` | Принудительный stop, partial result |
| `session.cost_usd >= $1` | Circuit breaker |
| `session.agent_steps >= 15` | Circuit breaker |
| JSON validation failed 2x подряд | STOP с ошибкой |

## Agent Loop: Normative Checker

### Реализация (per parameter)

```python
async def check_single_parameter(param: Parameter, session: SessionState) -> CheckResult:
    messages = [
        {"role": "system", "content": CHECKER_SYSTEM_PROMPT},
        {"role": "user", "content": format_parameter_for_check(param)},
    ]
    tools = [search_norms, get_norm_chunk, compare_values]

    for iteration in range(MAX_SEARCH_ITERATIONS):  # MAX_SEARCH_ITERATIONS = 3
        check_budget(session)

        response = await client.chat.completions.create(
            model="anthropic/claude-sonnet",
            messages=messages,
            tools=tool_schemas(tools),
            temperature=0.0,
        )
        session.agent_steps += 1
        track_usage(session, response)

        if response.choices[0].finish_reason == "stop":
            result = parse_check_result(response)
            # Верификация: chunk_id должен существовать
            if not verify_chunk_id(result.source_chunk_id):
                messages.append({"role": "user", "content": "Указанный chunk_id не найден. Повтори поиск."})
                continue
            return result

        for tool_call in response.choices[0].message.tool_calls:
            result = execute_tool(tool_call, session)
            messages.append(tool_call_result(tool_call, result))

        messages.append(response.choices[0].message)

    # Если не получили результат за 3 итерации
    return CheckResult(parameter=param, status="MANUAL", confidence=0.0,
                       explanation="Не удалось найти применимый норматив")
```

### Верификация chunk_id

Каждый `source_chunk_id` в ответе агента проверяется на существование в metadata store. Если chunk_id не найден — это галлюцинация, ответ отклоняется и агент делает повторный поиск.

## Retry policy

| Компонент | Retry | Backoff | Условие retry |
| --------- | ----- | ------- | ------------- |
| OpenRouter API call | 3 | Exponential (2s, 4s, 8s) | HTTP 429, 5xx, timeout |
| JSON validation | 2 | Немедленно | Невалидный JSON в ответе LLM |
| chunk_id verification | 1 | Немедленно | chunk_id не найден в metadata |
