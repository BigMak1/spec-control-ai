# C4 Context: SpecControl AI

## Описание

Диаграмма показывает систему SpecControl AI как единый блок, её пользователей и внешние зависимости.
Уровень C4 Level 1 — система как чёрный ящик: кто с ней взаимодействует и на какие внешние сервисы она опирается.

## Диаграмма

```mermaid
graph TB
    User["Специалист по нормоконтролю<br/>(Пользователь)"]

    subgraph boundary ["Граница системы"]
        System["SpecControl AI<br/>Проверка технической документации<br/>на соответствие нормативам"]
    end

    OpenRouter["OpenRouter API<br/>(Claude Sonnet LLM)"]
    LangFuse["LangFuse<br/>(self-hosted, Docker)<br/>Observability"]

    User -->|"Загружает PDF/DOCX<br/>Получает отчёт"| System
    System -->|"LLM запросы:<br/>извлечение параметров,<br/>проверка нормативов,<br/>генерация отчёта"| OpenRouter
    System -->|"Traces, metrics,<br/>cost tracking"| LangFuse
```

## Внешние зависимости

| Сервис | Тип связи | Критичность | Fallback |
| ------ | --------- | ----------- | -------- |
| OpenRouter API | HTTP REST | Критичен (без LLM система не работает) | Retry 3x exp. backoff, сообщение пользователю |
| LangFuse | HTTP | Некритичен (observability) | Система работает без LangFuse, логи пишутся локально |
