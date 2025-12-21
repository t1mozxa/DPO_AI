# S03-S04 – eda_cli: мини-EDA для CSV + HTTP-сервис качества данных

Небольшое CLI-приложение для базового анализа CSV-файлов и HTTP-сервис для оценки качества данных.
Используется в рамках Семинаров 03 и 04 курса «Инженерия ИИ».

## Требования

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) установлен в систему

## Инициализация проекта

В корне проекта:

```bash
uv sync
```

Эта команда:

- создаст виртуальное окружение `.venv`;
- установит зависимости из `pyproject.toml`;
- установит сам проект `eda-cli` в окружение.

## CLI-приложение

### Краткий обзор

```bash
uv run eda-cli overview data/example.csv
```

Параметры:

- `--sep` – разделитель (по умолчанию `,`);
- `--encoding` – кодировка (по умолчанию `utf-8`).

### Полный EDA-отчёт

```bash
uv run eda-cli report data/example.csv --out-dir reports
```

Доступные параметры:

- `--max-hist-columns` – сколько числовых колонок включать в набор гистограмм (по умолчанию: 6);
- `--top-k-categories` – сколько top-значений выводить для категориальных признаков (по умолчанию: 5);
- `--report-title` – заголовок отчёта (по умолчанию: "EDA-отчёт");
- `--min-missing-share` – порог доли пропусков, выше которого колонка считается проблемной и попадает в отдельный список в отчёте (по умолчанию: 0.1);
- `--json-summary` – сохранить JSON-сводку по датасету.


В результате в каталоге `reports/` появятся:

- `report.md` – основной отчёт в Markdown;
- `summary.csv` – таблица по колонкам;
- `missing.csv` – пропуски по колонкам;
- `correlation.csv` – корреляционная матрица (если есть числовые признаки);
- `top_categories/*.csv` – top-k категорий по строковым признакам;
- `hist_*.png` – гистограммы числовых колонок;
- `missing_matrix.png` – визуализация пропусков;
- `correlation_heatmap.png` – тепловая карта корреляций.
- `summary.json` – JSON-сводка по датасету (если указана опция `--json-summary`).

## HTTP-сервис качества данных

Запуск HTTP-сервиса:

```bash
uv run uvicorn eda_cli.api:app --reload --port 8000
```

После запуска доступна документация API по адресу: http://localhost:8000/docs

### Доступные эндпоинты

#### `GET /health`
Системный эндпоинт для проверки состояния сервиса.

Пример запроса:
```bash
curl http://localhost:8000/health
```

#### `POST /quality`
Эндпоинт, который принимает агрегированные признаки датасета и возвращает оценку качества.

Пример запроса:
```bash
curl -X POST http://localhost:8000/quality \
  -H "Content-Type: application/json" \
  -d '{
    "n_rows": 1000,
    "n_cols": 10,
    "max_missing_share": 0.1,
    "numeric_cols": 5,
    "categorical_cols": 5
  }'
```

#### `POST /quality-from-csv`
Эндпоинт, который принимает CSV-файл и возвращает оценку качества данных на основе EDA-анализа.

Пример запроса:
```bash
curl -X POST http://localhost:8000/quality-from-csv \
  -F "file=@data/example.csv"
```

#### `POST /quality-flags-from-csv` (новый эндпоинт из HW03)
Эндпоинт, который принимает CSV-файл и возвращает полный набор флагов качества, включая те, что были добавлены в HW03:
- `has_constant_columns` – наличие колонок с постоянными значениями
- `has_high_cardinality_categoricals` – наличие категориальных колонок с высокой кардинальностью
- `has_many_zero_values` – наличие числовых колонок с большим количеством нулей
- `has_suspicious_id_duplicates` – наличие подозрительных дубликатов ID

Пример запроса:
```bash
curl -X POST http://localhost:8000/quality-flags-from-csv \
  -F "file=@data/example.csv"
```

#### `POST /summary-from-csv` (дополнительный эндпоинт)
Эндпоинт, который принимает CSV-файл и возвращает JSON-сводку по датасету, аналогично опции CLI `--json-summary`.
Включает информацию о размерах датасета, оценке качества и проблемных колонках.

Пример запроса:
```bash
curl -X POST http://localhost:8000/summary-from-csv \
  -F "file=@data/example.csv"
```

## Тесты

```bash
uv run pytest -q
```

## Дополнительно

Проект был протестирован не только на встроенном `data/example.csv`, но и на собственных данных:
- `spotify_tracks.csv` и `second.csv` (локальный файл, не включён в репозиторий).

Пример вызова с новыми опциями (включая `--json-summary`):

```bash
uv run eda-cli report data/example.csv --out-dir reports --max-hist-columns 10 --top-k-categories 10 --report-title "Мой EDA-отчёт" --min-missing-share 0.15 --json-summary
