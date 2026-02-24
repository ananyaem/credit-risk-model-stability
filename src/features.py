"""Feature engineering utilities for Home Credit data."""

import polars as pl


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
