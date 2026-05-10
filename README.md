
# Web-UI-Agent

## Запуск:

1. Добавить в `.env`:

   * `OPENAI_API_KEY=ваш_ключ`  (обязательно)
   * `BROWSER_USE_API_KEY=ваш_ключ` *(опционально, для browser-use API модели)*
   * `OPEN_ROUTER_MODEL=model` *(опционально, используется как default если не передан --model)*

---

2. Установить зависимости:
   `pip install -r requirements.txt`

---

3. Распаковать большие HAR записи из ZIP файлов в папке `benchmark_data`

---

4. Установить браузеры:

   * `python -m playwright install chromium`
   * `uvx browser-use install`

---

5. Запустить бенчмарк из корня проекта:

### 🔹 OpenRouter агент

```bash
python -m benchmark.benchmark_runner --agent openrouter --model deepseek/deepseek-chat-v3.1
```

Модель задаётся через `--model`
Если не указать — возьмётся из `.env`

---

### 🔹 Browser-use агент

1. **Сначала запустить Chrome с CDP:**

```bash
"C:\Program Files\Google\Chrome\Application\chrome.exe" ^
  --remote-debugging-port=9222 ^
  --user-data-dir="C:\temp\chrome-debug"
```

2. Затем запустить бенчмарк:

```bash
python -m benchmark.benchmark_runner ^
  --agent browser_use ^
  --model deepseek/deepseek-chat-v3.1 ^
---
```
### Ограничить количество сценариев (для теста)

```bash
--limit 3
```

---

## Результаты

Результаты каждого прогона записываются в:

```text
benchmark_results.json
```

Каждый новый запуск **добавляется**, а не перезаписывает файл.

В результатах сохраняется:

* агент и модель
* метрики
* подробный трейс действий

---

## Запись HAR файлов:

1. В `playwright_recording.py` вставить название файла для записи и ссылку на сайт, чтобы playwright его открыл.
2. Запустить скрипт, откроется браузер chromium.
3. Проделать действия, которые должен будет проделать агент.
4. Закрыть chromium, файл сохранится в папку benchmark_data.

---

## Чтение HAR файлов

1. В `playwright_reading.py` вставить название файла для записи и ссылку на сайт, чтобы playwright его открыл.
2. Запустить скрипт, откроется браузер.
3. Можно ходить по сайту в пределах записанных страниц.

---



