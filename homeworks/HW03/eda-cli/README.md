# S03 – eda_cli: мини-EDA для CSV

Небольшое CLI-приложение для базового анализа CSV-файлов.
Используется в рамках Семинара 03 курса «Инженерия ИИ».

## Требования

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) установлен в систему

## Инициализация проекта

В корне проекта (S03):

```bash
uv sync
```

Эта команда:

- создаст виртуальное окружение `.venv`;
- установит зависимости из `pyproject.toml`;
- установит сам проект `eda-cli` в окружение.

## Запуск CLI

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
```
