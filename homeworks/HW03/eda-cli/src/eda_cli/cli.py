from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from .core import (
    DatasetSummary,
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
)
from .viz import (
    plot_correlation_heatmap,
    plot_missing_matrix,
    plot_histograms_per_column,
    save_top_categories_tables,
)

app = typer.Typer(help="Мини-CLI для EDA CSV-файлов")


def _load_csv(
    path: Path,
    sep: str = ",",
    encoding: str = "utf-8",
) -> pd.DataFrame:
    if not path.exists():
        raise typer.BadParameter(f"Файл '{path}' не найден")
    try:
        return pd.read_csv(path, sep=sep, encoding=encoding)
    except Exception as exc:  # noqa: BLE001
        raise typer.BadParameter(f"Не удалось прочитать CSV: {exc}") from exc


@app.command()
def overview(
    path: str = typer.Argument(..., help="Путь к CSV-файлу."),
    sep: str = typer.Option(",", help="Разделитель в CSV."),
    encoding: str = typer.Option("utf-8", help="Кодировка файла."),
) -> None:
    """
    Напечатать краткий обзор датасета:
    - размеры;
    - типы;
    - простая табличка по колонкам.
    """
    df = _load_csv(Path(path), sep=sep, encoding=encoding)
    summary: DatasetSummary = summarize_dataset(df)
    summary_df = flatten_summary_for_print(summary)

    typer.echo(f"Строк: {summary.n_rows}")
    typer.echo(f"Столбцов: {summary.n_cols}")
    typer.echo("\nКолонки:")
    typer.echo(summary_df.to_string(index=False))


@app.command()
def report(
    path: str = typer.Argument(..., help="Путь к CSV-файлу."),
    out_dir: str = typer.Option("reports", help="Каталог для отчёта."),
    sep: str = typer.Option(",", help="Разделитель в CSV."),
    encoding: str = typer.Option("utf-8", help="Кодировка файла."),
    max_hist_columns: int = typer.Option(6, help="Максимум числовых колонок для гистограмм."),
    top_k_categories: int = typer.Option(5, help="Количество top-значений для категориальных признаков."),
    report_title: str = typer.Option("EDA-отчёт", help="Заголовок отчёта."),
    min_missing_share: float = typer.Option(0.1, help="Минимальная доля пропусков для включения в отчёт проблемных колонок."),
    json_summary: bool = typer.Option(False, help="Сохранить JSON-сводку по датасету"),
) -> None:
    """
    Сгенерировать полный EDA-отчёт:
    - текстовый overview и summary по колонкам (CSV/Markdown);
    - статистика пропусков;
    - корреляционная матрица;
    - top-k категорий по категориальным признакам;
    - картинки: гистограммы, матрица пропусков, heatmap корреляции.
    """
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    df = _load_csv(Path(path), sep=sep, encoding=encoding)

    # 1. Обзор
    summary = summarize_dataset(df)
    summary_df = flatten_summary_for_print(summary)
    missing_df = missing_table(df)
    corr_df = correlation_matrix(df)
    top_cats = top_categories(df, top_k=top_k_categories)

    # 2. Качество в целом
    quality_flags = compute_quality_flags(summary, missing_df)

    # 3. Сохраняем табличные артефакты
    summary_df.to_csv(out_root / "summary.csv", index=False)
    if not missing_df.empty:
        missing_df.to_csv(out_root / "missing.csv", index=True)
    if not corr_df.empty:
        corr_df.to_csv(out_root / "correlation.csv", index=True)
    save_top_categories_tables(top_cats, out_root / "top_categories")

    # 4. Markdown-отчёт
    md_path = out_root / "report.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# {report_title}\n\n")
        f.write(f"Исходный файл: `{Path(path).name}`\n\n")
        f.write(f"Строк: **{summary.n_rows}**, столбцов: **{summary.n_cols}**\n\n")

        f.write("## Качество данных (эвристики)\n\n")
        f.write(f"- Оценка качества: **{quality_flags['quality_score']:.2f}**\n")
        f.write(f"- Макс. доля пропусков по колонке: **{quality_flags['max_missing_share']:.2%}**\n")
        f.write(f"- Слишком мало строк: **{quality_flags['too_few_rows']}**\n")
        f.write(f"- Слишком много колонок: **{quality_flags['too_many_columns']}**\n")
        f.write(f"- Слишком много пропусков: **{quality_flags['too_many_missing']}**\n")
        f.write(f"- Наличие константных колонок: **{quality_flags['has_constant_columns']}**\n")
        f.write(f"- Наличие категориальных признаков с высокой кардинальностью: **{quality_flags['has_high_cardinality_categoricals']}**\n")
        f.write(f"- Наличие числовых колонок с большим количеством нулей: **{quality_flags['has_many_zero_values']}**\n")
        f.write(f"- Наличие подозрительных дубликатов ID: **{quality_flags['has_suspicious_id_duplicates']}**\n\n")
        
        f.write(f"## Параметры отчёта\n\n")
        f.write(f"- Минимальная доля пропусков для проблемных колонок: **{min_missing_share:.2%}**\n")
        f.write(f"- Количество top-категорий: **{top_k_categories}**\n")
        f.write(f"- Максимум колонок для гистограмм: **{max_hist_columns}**\n\n")

        f.write("## Колонки\n\n")
        f.write("См. файл `summary.csv`.\n\n")

        f.write("## Пропуски\n\n")
        if missing_df.empty:
            f.write("Пропусков нет или датасет пуст.\n\n")
        else:
            f.write("См. файлы `missing.csv` и `missing_matrix.png`.\n\n")

        f.write("## Корреляция числовых признаков\n\n")
        if corr_df.empty:
            f.write("Недостаточно числовых колонок для корреляции.\n\n")
        else:
            f.write("См. `correlation.csv` и `correlation_heatmap.png`.\n\n")

        f.write("## Категориальные признаки\n\n")
        if not top_cats:
            f.write("Категориальные/строковые признаки не найдены.\n\n")
        else:
            f.write("См. файлы в папке `top_categories/`.\n\n")

        f.write("## Гистограммы числовых колонок\n\n")
        f.write("См. файлы `hist_*.png`.\n")

    # 5. Картинки
    plot_histograms_per_column(df, out_root, max_columns=max_hist_columns)
    plot_missing_matrix(df, out_root / "missing_matrix.png")
    plot_correlation_heatmap(df, out_root / "correlation_heatmap.png")
    
    # 6. JSON-сводка (опционально)
    if json_summary:
        import json
        # Создаём компактную сводку
        json_summary_data = {
            "n_rows": summary.n_rows,
            "n_cols": summary.n_cols,
            "quality_score": quality_flags["quality_score"],
            "problematic_columns": []
        }
        
        # Добавляем информацию о проблемных колонках
        if quality_flags["too_many_missing"]:
            for col in summary.columns:
                if col.missing_share > 0.5:
                    json_summary_data["problematic_columns"].append({
                        "name": col.name,
                        "issue": "too_many_missing",
                        "missing_share": col.missing_share
                    })
        
        if quality_flags["has_constant_columns"]:
            for col_name in quality_flags["constant_columns"]:
                json_summary_data["problematic_columns"].append({
                    "name": col_name,
                    "issue": "constant_column",
                    "unique_values": 1
                })
        
        if quality_flags["has_high_cardinality_categoricals"]:
            for col_name in quality_flags["high_cardinality_categoricals"]:
                col = next(c for c in summary.columns if c.name == col_name)
                cardinality_ratio = col.unique / summary.n_rows if summary.n_rows > 0 else 0
                json_summary_data["problematic_columns"].append({
                    "name": col_name,
                    "issue": "high_cardinality",
                    "cardinality_ratio": cardinality_ratio,
                    "unique_count": col.unique
                })
        
        if quality_flags["has_many_zero_values"]:
            for col_name in quality_flags["many_zero_columns"]:
                col = next(c for c in summary.columns if c.name == col_name)
                zero_ratio = col.zeros / col.non_null if col.non_null > 0 else 0
                json_summary_data["problematic_columns"].append({
                    "name": col_name,
                    "issue": "many_zero_values",
                    "zero_ratio": zero_ratio,
                    "zero_count": col.zeros
                })
        
        if quality_flags["has_suspicious_id_duplicates"]:
            for col_name in quality_flags["suspicious_id_columns"]:
                col = next(c for c in summary.columns if c.name == col_name)
                json_summary_data["problematic_columns"].append({
                    "name": col_name,
                    "issue": "suspicious_id_duplicates",
                    "unique_count": col.unique
                })
        
        # Сохраняем JSON-сводку
        json_path = out_root / "summary.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_summary_data, f, indent=2, ensure_ascii=False)
        
        typer.echo(f"- JSON-сводка: {json_path}")

    typer.echo(f"Отчёт сгенерирован в каталоге: {out_root}")
    typer.echo(f"- Основной markdown: {md_path}")
    typer.echo("- Табличные файлы: summary.csv, missing.csv, correlation.csv, top_categories/*.csv")
    typer.echo("- Графики: hist_*.png, missing_matrix.png, correlation_heatmap.png")
    if json_summary:
        typer.echo("- JSON-файл: summary.json")


if __name__ == "__main__":
    app()
