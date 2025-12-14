from __future__ import annotations

import pandas as pd

from eda_cli.core import (
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [10, 20, 30, None],
            "height": [140, 150, 160, 170],
            "city": ["A", "B", "A", None],
        }
    )


def test_summarize_dataset_basic():
    df = _sample_df()
    summary = summarize_dataset(df)

    assert summary.n_rows == 4
    assert summary.n_cols == 3
    assert any(c.name == "age" for c in summary.columns)
    assert any(c.name == "city" for c in summary.columns)

    summary_df = flatten_summary_for_print(summary)
    assert "name" in summary_df.columns
    assert "missing_share" in summary_df.columns


def test_missing_table_and_quality_flags():
    df = _sample_df()
    missing_df = missing_table(df)

    assert "missing_count" in missing_df.columns
    assert missing_df.loc["age", "missing_count"] == 1

    summary = summarize_dataset(df)
    flags = compute_quality_flags(summary, missing_df)
    assert 0.0 <= flags["quality_score"] <= 1.0


def test_correlation_and_top_categories():
    df = _sample_df()
    corr = correlation_matrix(df)
    # корреляция между age и height существует
    assert "age" in corr.columns or corr.empty is False

    top_cats = top_categories(df, max_columns=5, top_k=2)
    assert "city" in top_cats
    city_table = top_cats["city"]
    assert "value" in city_table.columns
    assert len(city_table) <= 2


def test_quality_flags_new_features():
    # Создаем DataFrame с константной колонкой для проверки has_constant_columns
    df_with_constant = pd.DataFrame({
        "age": [10, 20, 30, 40],
        "height": [140, 150, 160, 170],
        "city": ["A", "B", "C", "D"],
        "constant_col": [5, 5, 5, 5],  # константная колонка
        "user_id": [1, 2, 3, 3]  # дубликаты ID
    })
    
    summary = summarize_dataset(df_with_constant)
    missing_df = missing_table(df_with_constant)
    flags = compute_quality_flags(summary, missing_df)
    
    assert flags["has_constant_columns"] is True
    assert "constant_col" in flags["constant_columns"]
    
    assert flags["has_suspicious_id_duplicates"] is True
    assert "user_id" in flags["suspicious_id_columns"]


def test_quality_flags_many_zeros():
    # Создаем DataFrame с колонкой, в которой много нулей
    df_with_many_zeros = pd.DataFrame({
        "normal_col": [1, 2, 3, 4, 5],
        "zeros_col": [0, 0, 0, 0, 0]  # 5 из 5 значений - нули (100%)
    })
    
    summary = summarize_dataset(df_with_many_zeros)
    missing_df = missing_table(df_with_many_zeros)
    flags = compute_quality_flags(summary, missing_df)
    
    assert flags["has_many_zero_values"] is True
    assert "zeros_col" in flags["many_zero_columns"]


def test_quality_flags_high_cardinality():
    # Создаем DataFrame с категориальной колонкой с высокой кардинальностью
    # (>50% уникальных значений от общего числа строк)
    df_with_high_cardinality = pd.DataFrame({
        "normal_col": [1, 2, 3, 4],
        "high_cardinality_col": ["a", "b", "c", "d"]  # тут 4 уникальных значения из 4 строк (100%)
    })
    
    summary = summarize_dataset(df_with_high_cardinality)
    missing_df = missing_table(df_with_high_cardinality)
    flags = compute_quality_flags(summary, missing_df)
    
    assert flags["has_high_cardinality_categoricals"] is True
    assert "high_cardinality_col" in flags["high_cardinality_categoricals"]
