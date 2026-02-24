"""Feature engineering utilities for Home Credit data."""

from __future__ import annotations

import polars as pl

NUMERIC_DTYPES = frozenset({
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Float32, pl.Float64,
})
STRING_DTYPES = frozenset({pl.String, pl.Utf8, pl.Categorical})


def handle_dates(df: pl.DataFrame) -> pl.DataFrame:
    """
    Transform date and year columns into numeric features relative to date_decision.

    - Suffix 'D' columns: years before decision = (col - date_decision).total_days() / -365
    - Columns containing 'year': difference from decision year
    - Drops date_decision and MONTH afterwards.
    """
    if "date_decision" not in df.columns:
        raise ValueError("DataFrame must contain 'date_decision' column")

    date_d_cols = [
        c for c in df.columns
        if c.endswith("D") and c != "date_decision"
    ]
    year_cols = [
        c for c in df.columns
        if "year" in c.lower() and c not in date_d_cols and c != "date_decision"
    ]

    exprs = []

    for col in date_d_cols:
        exprs.append(
            ((pl.col(col) - pl.col("date_decision")).dt.total_days() / -365)
            .cast(pl.Float32)
            .alias(col)
        )

    for col in year_cols:
        exprs.append(
            (pl.col(col) - pl.col("date_decision").dt.year())
            .cast(pl.Float32)
            .alias(col)
        )

    if exprs:
        df = df.with_columns(exprs)

    cols_to_drop = [c for c in ("date_decision", "MONTH") if c in df.columns]
    return df.drop(cols_to_drop)


def create_domain_ratios(df: pl.DataFrame) -> pl.DataFrame:
    """
    Create domain-specific ratio features from static_0 columns.

    Computes:
      loan_burden_ratio        = price_1097A / annuity_780A
      disbursed_credit_ratio   = disbursedcredamount_1113A / credamount_770A
      debt_credit_ratio        = totaldebt_9A / (1 + credamount_770A)
      eir_credit_ratio         = eir_270L / credamount_770A
    """
    COL_MAP = {
        "price": "price_1097A",
        "annuity": "annuity_780A",
        "disbursed": "disbursedcredamount_1113A",
        "credit_amount": "credamount_770A",
        "total_debt": "totaldebt_9A",
        "eir": "eir_270L",
    }

    available = {k: v for k, v in COL_MAP.items() if v in df.columns}
    missing = set(COL_MAP.values()) - set(df.columns)
    if missing:
        print(f"[create_domain_ratios] columns not found, skipping related ratios: {sorted(missing)}")

    ratio_exprs = []

    if {"price", "annuity"} <= available.keys():
        ratio_exprs.append(
            (pl.col(available["price"]) / pl.col(available["annuity"]))
            .cast(pl.Float32)
            .alias("loan_burden_ratio")
        )

    if {"disbursed", "credit_amount"} <= available.keys():
        ratio_exprs.append(
            (pl.col(available["disbursed"]) / pl.col(available["credit_amount"]))
            .cast(pl.Float32)
            .alias("disbursed_credit_ratio")
        )

    if {"total_debt", "credit_amount"} <= available.keys():
        ratio_exprs.append(
            (pl.col(available["total_debt"]) / (1 + pl.col(available["credit_amount"])))
            .cast(pl.Float32)
            .alias("debt_credit_ratio")
        )

    if {"eir", "credit_amount"} <= available.keys():
        ratio_exprs.append(
            (pl.col(available["eir"]) / pl.col(available["credit_amount"]))
            .cast(pl.Float32)
            .alias("eir_credit_ratio")
        )

    if ratio_exprs:
        df = df.with_columns(ratio_exprs)

    return df


def _build_agg_exprs(
    df: pl.DataFrame,
    skip: set[str],
) -> tuple[list[pl.Expr], dict[str, int]]:
    """Classify columns and return (agg_exprs, stats).

    Shared by aggregate_depth1 and aggregate_depth2.
    """
    n_rows = df.height

    numeric_cols: list[str] = []
    amount_cols: list[str] = []
    string_mode_cols: list[str] = []
    cat_count_map: dict[str, list[str]] = {}

    for col in df.columns:
        if col in skip:
            continue
        dtype = df[col].dtype

        if dtype in NUMERIC_DTYPES:
            numeric_cols.append(col)
            if col.endswith("A"):
                amount_cols.append(col)

        elif dtype in STRING_DTYPES:
            n_uniq = df[col].n_unique()
            null_rate = df[col].null_count() / n_rows if n_rows > 0 else 1.0

            if n_uniq <= 200:
                string_mode_cols.append(col)

            if n_uniq <= 10 and null_rate < 0.9:
                cat_count_map[col] = df[col].drop_nulls().unique().to_list()

    agg_exprs: list[pl.Expr] = []

    for col in numeric_cols:
        agg_exprs.extend([
            pl.col(col).mean().alias(f"{col}_mean"),
            pl.col(col).max().alias(f"{col}_max"),
            pl.col(col).min().alias(f"{col}_min"),
            pl.col(col).first().alias(f"{col}_first"),
            pl.col(col).last().alias(f"{col}_last"),
            pl.col(col).std().alias(f"{col}_std"),
        ])

    for col in amount_cols:
        agg_exprs.append(
            (pl.col(col).std() / (pl.col(col).mean().abs() + 1e-9))
            .alias(f"{col}_cv")
        )

    for col in string_mode_cols:
        agg_exprs.extend([
            pl.col(col).drop_nulls().mode().first().alias(f"{col}_mode"),
            pl.col(col).n_unique().alias(f"{col}_nunique"),
        ])

    for col, vals in cat_count_map.items():
        for val in vals:
            safe = str(val).replace(" ", "_").replace("/", "_")
            agg_exprs.append(
                (pl.col(col) == val).sum().alias(f"{col}_{safe}_count")
            )

    stats = {
        "numeric": len(numeric_cols),
        "amount_cv": len(amount_cols),
        "string_mode": len(string_mode_cols),
        "cat_count": len(cat_count_map),
    }
    return agg_exprs, stats


# ─────────────────────────────────────────────────────────────────
#  Public aggregation functions
# ─────────────────────────────────────────────────────────────────

def aggregate_depth1(
    df: pl.DataFrame,
    group_col: str = "case_id",
) -> pl.DataFrame:
    """
    Aggregate a depth-1 table by *group_col* after sorting by num_group1.

    Aggregation rules
    -----------------
    * Numeric columns          → mean, max, min, first, last, std
    * Amount columns (suffix A)→ additionally coefficient of variation (std / |mean|)
    * String cols  (≤200 uniq) → mode, n_unique
    * Categoricals (≤10 uniq, <0.9 null rate) → per-value occurrence counts
    """
    skip = {group_col, "num_group1", "num_group2"}

    if "num_group1" in df.columns:
        df = df.sort(group_col, "num_group1")

    agg_exprs, stats = _build_agg_exprs(df, skip)

    if not agg_exprs:
        return df.select(group_col).unique()

    result = df.group_by(group_col).agg(agg_exprs)

    print(
        f"[aggregate_depth1] {stats['numeric']} numeric "
        f"({stats['amount_cv']} amount w/ CV), "
        f"{stats['string_mode']} string (mode/nunique), "
        f"{stats['cat_count']} categorical (value counts) "
        f"→ {result.shape[1] - 1} features"
    )
    return result


def aggregate_depth2(df: pl.DataFrame) -> pl.DataFrame:
    """
    Two-pass aggregation for depth-2 tables.

    Pass 1: group by (case_id, num_group1) — collapses the num_group2 dimension.
    Pass 2: group by case_id               — collapses the num_group1 dimension.

    Both passes use the same aggregation rules as aggregate_depth1.
    """
    # ── pass 1: collapse num_group2 ─────────────────────────────────
    skip1 = {"case_id", "num_group1", "num_group2"}

    if "num_group2" in df.columns:
        df = df.sort("case_id", "num_group1", "num_group2")

    agg1, stats1 = _build_agg_exprs(df, skip1)

    if not agg1:
        return df.select("case_id").unique()

    pass1 = df.group_by(["case_id", "num_group1"]).agg(agg1)

    print(
        f"[aggregate_depth2] pass 1 (by case_id, num_group1): "
        f"{stats1['numeric']} numeric, {stats1['string_mode']} string "
        f"→ {pass1.shape[1] - 2} features, {pass1.height:,} rows"
    )

    # ── pass 2: collapse num_group1 ─────────────────────────────────
    skip2 = {"case_id", "num_group1"}
    pass1 = pass1.sort("case_id", "num_group1")

    agg2, stats2 = _build_agg_exprs(pass1, skip2)

    if not agg2:
        return pass1.select("case_id").unique()

    result = pass1.group_by("case_id").agg(agg2)

    print(
        f"[aggregate_depth2] pass 2 (by case_id): "
        f"{stats2['numeric']} numeric, {stats2['string_mode']} string "
        f"→ {result.shape[1] - 1} features, {result.height:,} rows"
    )
    return result
