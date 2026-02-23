"""
Polars utilities for loading and preprocessing Home Credit data.
"""

import polars as pl
from pathlib import Path


def load_table_group(
    data_path: str | Path,
    table_name: str,
    split: str = "train",
    file_format: str = "parquet"
) -> pl.DataFrame:
    """
    Load all parquet/csv files for a given table group and concatenate them vertically.
    
    Args:
        data_path: Root path to the data directory
        table_name: Name of the table group (e.g., 'base', 'static_0', 'static_cb_0')
        split: Either 'train' or 'test'
        file_format: Either 'parquet' or 'csv'
    
    Returns:
        Concatenated DataFrame from all matching files
    
    Example:
        >>> df = load_table_group("/data", "static_0", split="train")
        # Loads train_static_0_0.parquet, train_static_0_1.parquet, etc.
    """
    data_path = Path(data_path)
    
    if file_format == "parquet":
        subdir = "parquet_files" / Path(split)
        ext = ".parquet"
        read_fn = pl.read_parquet
    else:
        subdir = "csv_files" / Path(split)
        ext = ".csv"
        read_fn = pl.read_csv
    
    full_dir = data_path / subdir
    pattern = f"{split}_{table_name}"
    
    matching_files = sorted(full_dir.glob(f"{pattern}*{ext}"))
    
    if not matching_files:
        raise FileNotFoundError(
            f"No files found matching pattern '{pattern}*{ext}' in {full_dir}"
        )
    
    dfs = [read_fn(f) for f in matching_files]
    
    if len(dfs) == 1:
        return dfs[0]
    
    return pl.concat(dfs, how="vertical_relaxed")


def downcast_dtypes(df: pl.DataFrame) -> pl.DataFrame:
    """
    Downcast float64 to float32 and int64 to int32 to reduce memory usage.
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with downcasted numeric types
    """
    exprs = []
    for col in df.columns:
        dtype = df[col].dtype
        if dtype == pl.Float64:
            exprs.append(pl.col(col).cast(pl.Float32))
        elif dtype == pl.Int64:
            exprs.append(pl.col(col).cast(pl.Int32))
        else:
            exprs.append(pl.col(col))
    
    return df.select(exprs)


def drop_high_missing_cols(
    df: pl.DataFrame,
    threshold: float = 0.98
) -> pl.DataFrame:
    """
    Drop columns where the missing rate exceeds the threshold.
    
    Args:
        df: Input DataFrame
        threshold: Maximum allowed missing rate (default 0.98 = 98%)
    
    Returns:
        DataFrame with high-missing columns removed
    """
    n_rows = df.height
    if n_rows == 0:
        return df
    
    null_rates = df.null_count() / n_rows
    cols_to_keep = [
        col for col in df.columns
        if null_rates[col][0] <= threshold
    ]
    
    dropped_cols = set(df.columns) - set(cols_to_keep)
    if dropped_cols:
        print(f"Dropped {len(dropped_cols)} columns with >{threshold*100:.0f}% missing: {sorted(dropped_cols)}")
    
    return df.select(cols_to_keep)


def drop_high_cardinality_string_cols(
    df: pl.DataFrame,
    max_unique: int = 10_000
) -> pl.DataFrame:
    """
    Drop string columns with more than max_unique unique values.
    
    Args:
        df: Input DataFrame
        max_unique: Maximum allowed unique values for string columns
    
    Returns:
        DataFrame with high-cardinality string columns removed
    """
    cols_to_drop = []
    
    for col in df.columns:
        if df[col].dtype == pl.String or df[col].dtype == pl.Utf8:
            n_unique = df[col].n_unique()
            if n_unique > max_unique:
                cols_to_drop.append(col)
    
    if cols_to_drop:
        print(f"Dropped {len(cols_to_drop)} string columns with >{max_unique} unique values: {cols_to_drop}")
        return df.drop(cols_to_drop)
    
    return df


def preprocess_table(
    df: pl.DataFrame,
    missing_threshold: float = 0.98,
    max_string_cardinality: int = 10_000,
    downcast: bool = True
) -> pl.DataFrame:
    """
    Apply all preprocessing steps to a table.
    
    Args:
        df: Input DataFrame
        missing_threshold: Max missing rate before dropping column
        max_string_cardinality: Max unique values for string columns
        downcast: Whether to downcast numeric types
    
    Returns:
        Preprocessed DataFrame
    """
    if downcast:
        df = downcast_dtypes(df)
    
    df = drop_high_missing_cols(df, threshold=missing_threshold)
    df = drop_high_cardinality_string_cols(df, max_unique=max_string_cardinality)
    
    return df


def get_table_info(df: pl.DataFrame) -> dict:
    """
    Get summary information about a DataFrame.
    
    Returns dict with shape, memory usage, dtypes, and missing rates.
    """
    n_rows, n_cols = df.shape
    
    null_counts = df.null_count()
    missing_rates = {col: null_counts[col][0] / n_rows for col in df.columns}
    
    dtype_counts = {}
    for col in df.columns:
        dtype_str = str(df[col].dtype)
        dtype_counts[dtype_str] = dtype_counts.get(dtype_str, 0) + 1
    
    return {
        "shape": (n_rows, n_cols),
        "estimated_memory_mb": df.estimated_size("mb"),
        "dtype_counts": dtype_counts,
        "columns_with_high_missing": [
            col for col, rate in missing_rates.items() if rate > 0.5
        ],
    }
