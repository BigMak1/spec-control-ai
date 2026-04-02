# Spec: Retriever

## Источники данных

| Документ | Формат | Примерный объём |
| -------- | ------ | --------------- |
| ПУЭ 7-е изд. (выбранные главы) | PDF | ~50-100 стр. |
| ГОСТ Р 50571 (выбранные части) | PDF | ~30-50 стр. |
| СП 76.13330 | PDF | ~40-60 стр. |
| Дополнительно 1-2 документа | PDF | ~30-50 стр. |

Все документы — русскоязычные, текстовые PDF (не сканы).

## Индексация (offline, one-time)

### Chunking стратегия

- **Метод:** семантический — по пунктам/разделам норматива, не механическое разбиение
- **Chunk size:** 500-800 tokens
- **Overlap:** ~100 tokens (конец предыдущего → начало следующего)
- **Заголовок:** включается в каждый chunk для контекста
- **Инструмент парсинга:** PyMuPDF для извлечения текста из PDF

### Embedding

- **Модель:** intfloat/multilingual-e5-large
- **Запуск:** локально, CPU
- **Размерность вектора:** 1024
- **Время индексации:** несколько минут для 3-5 документов

### Хранение

- **FAISS index:** файл `.faiss`, тип `IndexFlatIP` (inner product, после L2-нормализации = cosine similarity)
- **Metadata:** JSON-файл, маппинг `index_position → chunk_metadata`

### Chunk metadata schema

```json
{
  "chunk_id": "pue_7_1_34_002",
  "norm_doc": "ПУЭ 7-е изд.",
  "section": "7.1.34",
  "title": "Сечения кабелей",
  "page": 142,
  "text": "Полный текст чанка...",
  "version": "2003",
  "status": "действующий"
}
```

## Поиск (runtime)

### API

**search_norms(query: str, top_k: int = 5, filter_doc: str | None = None) → List[ChunkResult]**

1. Embedding запроса через e5-large (~200ms CPU)
2. FAISS similarity search (top_k, ~1ms)
3. Если `filter_doc` указан — post-filtering по `norm_doc` в metadata
4. Возврат: top-K chunks с score и полным metadata

**get_norm_chunk(chunk_id: str) → ChunkWithContext**

1. Lookup в metadata по chunk_id
2. Возврат: текст чанка + metadata + тексты соседних чанков (prev/next) для контекста

### Пороги

| Параметр | Значение | Действие при нарушении |
| -------- | -------- | ---------------------- |
| Min relevance score | 0.3 (cosine sim) | Чанки ниже порога не возвращаются |
| Max search iterations (agent) | 3 | Agent stop, status=MANUAL |

## Ограничения

- Reranking не используется в PoC (один этап retrieval)
- Нет real-time обновления индекса — только offline переиндексация
- При добавлении новых нормативов — полная переиндексация (приемлемо для 3-5 документов)
- Embedding модель не fine-tuned под нормативную лексику
