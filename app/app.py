"""Credit Risk Model Stability — Streamlit Dashboard.

Run from the project root:
    streamlit run app/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import streamlit as st
from scipy import stats as sp_stats
from sklearn.metrics import roc_auc_score

from src.data_processing import load_table_group, preprocess_table
from src.metrics import gini_stability

# ── Paths ────────────────────────────────────────────────────────
DATA_PATH = ROOT / "data"
ARTIFACTS_PATH = ROOT / "artifacts"

# ── Streamlit config ─────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk Stability",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Palette
BLUE = "#4C72B0"
ORANGE = "#DD8452"
GREEN = "#55A868"
RED = "#C44E52"
GREY = "#8C8C8C"
PURPLE = "#8172B3"

# ═════════════════════════════════════════════════════════════════
#  Cached data loaders
# ═════════════════════════════════════════════════════════════════


@st.cache_data(show_spinner="Loading training data ...")
def load_train_data() -> pd.DataFrame:
    """Load processed features if available, else base + static_0."""
    processed = DATA_PATH / "processed" / "train_final.parquet"
    if processed.exists():
        return pl.read_parquet(processed).to_pandas()

    base = load_table_group(DATA_PATH, "base", split="train")
    try:
        static_0 = preprocess_table(
            load_table_group(DATA_PATH, "static_0", split="train")
        )
        df = base.join(static_0, on="case_id", how="left")
    except FileNotFoundError:
        df = base
    return df.to_pandas()


@st.cache_data(show_spinner="Loading OOF predictions ...")
def load_oof_predictions() -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Load tuned OOF and ensemble OOF parquets."""
    tuned_path = ARTIFACTS_PATH / "tuned" / "tuned_oof.parquet"
    ens_path = ARTIFACTS_PATH / "tuned" / "ensemble_oof.parquet"

    if not tuned_path.exists():
        return None, None
    tuned = pl.read_parquet(tuned_path).to_pandas()
    ens = (
        pl.read_parquet(ens_path).to_pandas() if ens_path.exists() else None
    )
    return tuned, ens


# ═════════════════════════════════════════════════════════════════
#  Drift helpers
# ═════════════════════════════════════════════════════════════════

NUMERIC_PD_KINDS = {"i", "u", "f"}


def _ks_test(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    """KS test on two numeric series; returns (stat, p_value)."""
    x, y = a.dropna(), b.dropna()
    if len(x) < 10 or len(y) < 10:
        return np.nan, np.nan
    stat, pval = sp_stats.ks_2samp(x, y)
    return float(stat), float(pval)


def classify_drift(ks: float) -> tuple[str, str]:
    if np.isnan(ks):
        return "N/A", GREY
    if ks >= 0.15:
        return "High", RED
    if ks >= 0.05:
        return "Medium", ORANGE
    return "Low", GREEN


def _bin_weeks(
    weeks: np.ndarray, n_bins: int
) -> tuple[np.ndarray, list[str], int]:
    """Digitise WEEK_NUM into roughly equal-sized bins."""
    edges = np.unique(np.percentile(weeks, np.linspace(0, 100, n_bins + 1)))
    actual = len(edges) - 1
    labels = [f"W{int(edges[i])}\u2013{int(edges[i + 1])}" for i in range(actual)]
    idx = np.clip(np.digitize(weeks, edges, right=True), 1, actual)
    return idx, labels, actual


@st.cache_data(show_spinner="Computing drift scores ...")
def compute_drift_table(
    df: pd.DataFrame, week_col: str, n_bins: int
) -> pd.DataFrame:
    """KS statistic (earliest vs latest bin) for every numeric feature."""
    weeks = df[week_col].values
    bin_idx, _, actual = _bin_weeks(weeks, n_bins)

    num_cols = [
        c
        for c in df.columns
        if df[c].dtype.kind in NUMERIC_PD_KINDS and c not in ("case_id", "target", week_col)
    ]

    rows: list[dict] = []
    for col in num_cols:
        early = df.loc[bin_idx == 1, col]
        late = df.loc[bin_idx == actual, col]
        ks, pv = _ks_test(early, late)
        label, _ = classify_drift(ks)
        rows.append(
            {"Feature": col, "KS Statistic": ks, "p-value": pv, "Drift": label}
        )
    return pd.DataFrame(rows).sort_values("KS Statistic", ascending=False, na_position="last")


# ═════════════════════════════════════════════════════════════════
#  Stability helpers
# ═════════════════════════════════════════════════════════════════


def weekly_ginis(
    week_num: np.ndarray, target: np.ndarray, score: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-week Gini = 2*AUC - 1.  Returns (valid_weeks, gini_array)."""
    weeks = np.asarray(week_num)
    y, s = np.asarray(target), np.asarray(score)
    unique = np.sort(np.unique(weeks))
    ws, gs = [], []
    for w in unique:
        m = weeks == w
        if len(np.unique(y[m])) < 2:
            continue
        gs.append(2.0 * roc_auc_score(y[m], s[m]) - 1.0)
        ws.append(w)
    return np.array(ws), np.array(gs)


# ═════════════════════════════════════════════════════════════════
#  Page 1 — Data Drift
# ═════════════════════════════════════════════════════════════════


def page_data_drift() -> None:
    st.title("Data Drift")
    st.caption(
        "Compare feature distributions across WEEK_NUM bins to detect "
        "temporal drift that may degrade model stability."
    )

    df = load_train_data()

    if "WEEK_NUM" not in df.columns:
        st.error("`WEEK_NUM` column not found in training data.")
        return

    # Identify column types
    num_cols = sorted(
        c
        for c in df.columns
        if df[c].dtype.kind in NUMERIC_PD_KINDS
        and c not in ("case_id", "target", "WEEK_NUM")
    )
    str_cols = sorted(c for c in df.columns if df[c].dtype.kind in {"O", "S"})
    all_features = num_cols + str_cols

    if not all_features:
        st.warning("No feature columns found.")
        return

    # ── Controls ──────────────────────────────────────────────────
    left, right = st.columns([3, 1])
    with left:
        feature = st.selectbox("Feature", all_features, index=0)
    with right:
        n_bins = st.slider("Week bins", 2, 6, 3)

    weeks = df["WEEK_NUM"].values
    bin_idx, bin_labels, actual_bins = _bin_weeks(weeks, n_bins)
    is_numeric = feature in num_cols

    st.markdown("---")

    # ── Per-feature visualisation ─────────────────────────────────
    if is_numeric:
        st.subheader(f"`{feature}`  (numeric)")

        fig, (ax_hist, ax_trend) = plt.subplots(
            1, 2, figsize=(14, 4.5), gridspec_kw={"width_ratios": [3, 2]}
        )

        cmap = plt.cm.viridis(np.linspace(0.2, 0.85, actual_bins))
        for i in range(1, actual_bins + 1):
            vals = df.loc[bin_idx == i, feature].dropna()
            if vals.empty:
                continue
            lo, hi = np.nanpercentile(vals, [1, 99])
            clipped = vals[(vals >= lo) & (vals <= hi)]
            ax_hist.hist(
                clipped, bins=50, density=True, alpha=0.55,
                color=cmap[i - 1], label=bin_labels[i - 1], edgecolor="none",
            )
        ax_hist.set_title("Density per week bin", fontsize=11)
        ax_hist.set_ylabel("Density")
        ax_hist.legend(fontsize=8)

        means = [df.loc[bin_idx == i, feature].mean() for i in range(1, actual_bins + 1)]
        stds = [df.loc[bin_idx == i, feature].std() for i in range(1, actual_bins + 1)]
        x = np.arange(actual_bins)
        ax_trend.errorbar(
            x, means, yerr=stds, marker="o", capsize=5,
            color=BLUE, linewidth=1.5, markersize=6,
        )
        ax_trend.set_xticks(x)
        ax_trend.set_xticklabels(bin_labels, fontsize=9)
        ax_trend.set_title("Mean \u00b1 Std across bins", fontsize=11)

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        early = df.loc[bin_idx == 1, feature]
        late = df.loc[bin_idx == actual_bins, feature]
        ks_stat, ks_pval = _ks_test(early, late)
        drift_label, drift_colour = classify_drift(ks_stat)

        m1, m2, m3 = st.columns(3)
        m1.metric("KS Statistic", f"{ks_stat:.4f}" if not np.isnan(ks_stat) else "N/A")
        m2.metric("p-value", f"{ks_pval:.2e}" if not np.isnan(ks_pval) else "N/A")
        m3.markdown(
            f"**Drift severity:** "
            f"<span style='color:{drift_colour};font-weight:bold'>{drift_label}</span>",
            unsafe_allow_html=True,
        )

    else:
        st.subheader(f"`{feature}`  (categorical)")

        fig, ax = plt.subplots(figsize=(14, 4.5))
        groups: dict[str, pd.Series] = {}
        for i in range(1, actual_bins + 1):
            vc = df.loc[bin_idx == i, feature].value_counts(normalize=True)
            groups[bin_labels[i - 1]] = vc.head(15)
        prop_df = pd.DataFrame(groups).fillna(0)
        prop_df.plot.bar(ax=ax, width=0.8, edgecolor="none", alpha=0.85)
        ax.set_title("Top category proportions per week bin", fontsize=11)
        ax.set_ylabel("Proportion")
        ax.legend(fontsize=8)
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ── Drift summary table ───────────────────────────────────────
    st.markdown("---")
    st.subheader("Drift summary \u2014 all numeric features")
    st.caption("KS test between earliest and latest week bin, sorted by severity.")

    drift_df = compute_drift_table(df, "WEEK_NUM", n_bins)

    n_high = (drift_df["Drift"] == "High").sum()
    n_med = (drift_df["Drift"] == "Medium").sum()
    n_low = (drift_df["Drift"] == "Low").sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total features", len(drift_df))
    c2.metric("High drift", int(n_high))
    c3.metric("Medium drift", int(n_med))
    c4.metric("Low / None", int(n_low))

    def _drift_bg(val: str) -> str:
        return {
            "High": "background-color: #f8d7da",
            "Medium": "background-color: #fff3cd",
            "Low": "background-color: #d4edda",
        }.get(val, "")

    styled = (
        drift_df.head(100)
        .style.applymap(_drift_bg, subset=["Drift"])
        .format({"KS Statistic": "{:.4f}", "p-value": "{:.2e}"})
    )
    st.dataframe(styled, use_container_width=True, height=420)


# ═════════════════════════════════════════════════════════════════
#  Page 2 — Model Stability
# ═════════════════════════════════════════════════════════════════


def page_model_stability() -> None:
    st.title("Model Stability")
    st.caption(
        "Weekly Gini stability analysis: regression trend through weekly "
        "Gini coefficients, falling-rate penalty, and residual variance."
    )

    tuned_oof, ens_oof = load_oof_predictions()

    if tuned_oof is None:
        st.warning(
            "Tuned OOF predictions not found.  "
            "Run the training pipeline first to generate "
            "`artifacts/tuned/tuned_oof.parquet`."
        )
        return

    week_num = tuned_oof["WEEK_NUM"].values
    target = tuned_oof["target"].values

    # Discover available model columns
    MODEL_MAP: dict[str, str] = {
        "CatBoost": "oof_catboost",
        "LightGBM": "oof_lightgbm",
        "XGBoost": "oof_xgboost",
    }
    available: dict[str, str] = {
        k: v for k, v in MODEL_MAP.items() if v in tuned_oof.columns
    }

    if ens_oof is not None and "oof_ensemble" in ens_oof.columns:
        tuned_oof = tuned_oof.merge(
            ens_oof[["case_id", "oof_ensemble"]], on="case_id", how="left"
        )
        available["Ensemble"] = "oof_ensemble"

    if not available:
        st.error("No model prediction columns found in tuned_oof.parquet.")
        return

    # Compute stability for every model
    stab_results: dict[str, dict] = {}
    for name, col in available.items():
        stab_results[name] = gini_stability(week_num, target, tuned_oof[col].values)

    # ── Model selector ────────────────────────────────────────────
    default_idx = len(available) - 1
    selected = st.selectbox(
        "Model for detailed view", list(available.keys()), index=default_idx
    )
    sel_col = available[selected]
    sel_stab = stab_results[selected]

    # ── Metric cards ──────────────────────────────────────────────
    st.markdown("---")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Stability Score", f"{sel_stab['stability_score']:.4f}")
    k2.metric("Mean Gini", f"{sel_stab['mean_gini']:.4f}")
    k3.metric(
        "Falling Rate",
        f"{sel_stab['falling_rate']:.6f}",
        delta=f"slope = {sel_stab['slope']:.6f}",
        delta_color="inverse",
    )
    k4.metric("Std Residuals", f"{sel_stab['std_residuals']:.4f}")

    # ── Weekly Gini + residuals ───────────────────────────────────
    st.markdown("---")
    st.subheader("Weekly Gini curve")

    _, ginis = weekly_ginis(week_num, target, tuned_oof[sel_col].values)
    x = np.arange(len(ginis))

    if len(ginis) >= 2:
        slope, intercept = np.polyfit(x, ginis, 1)
    else:
        slope, intercept = 0.0, float(np.mean(ginis)) if len(ginis) else 0.0
    trend = slope * x + intercept
    residuals = ginis - trend

    fig, (ax_g, ax_r) = plt.subplots(
        2, 1, figsize=(14, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    ax_g.plot(x, ginis, "o-", color=BLUE, markersize=5, linewidth=1.3, label="Weekly Gini")
    ax_g.plot(
        x, trend, "--", color=ORANGE, linewidth=1.8,
        label=f"Trend (slope={slope:.5f})",
    )
    ax_g.axhline(
        sel_stab["mean_gini"], color=GREY, linestyle=":", linewidth=0.9,
        label=f"Mean = {sel_stab['mean_gini']:.4f}",
    )
    ax_g.fill_between(
        x, trend - sel_stab["std_residuals"], trend + sel_stab["std_residuals"],
        color=ORANGE, alpha=0.12, label="\u00b11 std residual",
    )
    ax_g.set_ylabel("Gini  (2 \u00b7 AUC \u2212 1)", fontsize=11)
    ax_g.set_title(
        f"{selected} \u2014 Weekly Gini  "
        f"(stability = {sel_stab['stability_score']:.4f})",
        fontsize=13,
    )
    ax_g.legend(fontsize=9, loc="lower left")
    ax_g.grid(axis="y", alpha=0.3)

    bar_colors = [RED if r < 0 else GREEN for r in residuals]
    ax_r.bar(x, residuals, color=bar_colors, alpha=0.7, edgecolor="none")
    ax_r.axhline(0, color="black", linewidth=0.5)
    ax_r.set_ylabel("Residual", fontsize=10)
    ax_r.set_xlabel("Week index", fontsize=11)
    ax_r.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ── Formula breakdown ─────────────────────────────────────────
    with st.expander("Stability metric formula"):
        st.latex(
            r"\text{stability} = \overline{G}"
            r" + 88 \cdot \min(0,\;\beta)"
            r" - 0.5 \cdot \sigma_{\varepsilon}"
        )
        breakdown = pd.DataFrame(
            [
                ("Mean Gini  (G\u0304)", f"{sel_stab['mean_gini']:.6f}"),
                ("Slope  (\u03b2)", f"{sel_stab['slope']:.6f}"),
                (
                    "Falling-rate penalty  (88 \u00b7 min(0, \u03b2))",
                    f"{88 * sel_stab['falling_rate']:.6f}",
                ),
                (
                    "Residual penalty  (\u22120.5 \u00b7 \u03c3\u03b5)",
                    f"{-0.5 * sel_stab['std_residuals']:.6f}",
                ),
                ("Stability score", f"{sel_stab['stability_score']:.6f}"),
            ],
            columns=["Component", "Value"],
        )
        st.table(breakdown)

    # ── Model-vs-ensemble comparison ──────────────────────────────
    st.markdown("---")
    st.subheader("Model vs Ensemble comparison")

    rows: list[dict] = []
    for name, col in available.items():
        s = stab_results[name]
        auc = roc_auc_score(target, tuned_oof[col].values)
        rows.append(
            {
                "Model": name,
                "AUC": auc,
                "Stability": s["stability_score"],
                "Mean Gini": s["mean_gini"],
                "Falling Rate": s["falling_rate"],
                "Std Residuals": s["std_residuals"],
            }
        )

    comp_df = pd.DataFrame(rows)
    styled_comp = (
        comp_df.style.format(
            {
                "AUC": "{:.6f}",
                "Stability": "{:.6f}",
                "Mean Gini": "{:.6f}",
                "Falling Rate": "{:.6f}",
                "Std Residuals": "{:.6f}",
            }
        )
        .highlight_max(
            subset=["AUC", "Stability", "Mean Gini"],
            props="background-color: #d4edda",
        )
        .highlight_min(
            subset=["Std Residuals"],
            props="background-color: #d4edda",
        )
    )
    st.dataframe(styled_comp, use_container_width=True)

    # ── All-models weekly Gini overlay ────────────────────────────
    st.subheader("Weekly Gini \u2014 all models")

    fig, ax = plt.subplots(figsize=(14, 5))
    palette = [BLUE, ORANGE, GREEN, RED, PURPLE]

    for i, (name, col) in enumerate(available.items()):
        _, g = weekly_ginis(week_num, target, tuned_oof[col].values)
        colour = palette[i % len(palette)]
        is_ens = name == "Ensemble"
        ax.plot(
            np.arange(len(g)), g, "o-", color=colour,
            markersize=4 if not is_ens else 5,
            linewidth=2.2 if is_ens else 1.2,
            alpha=1.0 if is_ens else 0.6,
            label=name,
        )

    ax.set_xlabel("Week index", fontsize=11)
    ax.set_ylabel("Gini  (2 \u00b7 AUC \u2212 1)", fontsize=11)
    ax.set_title("Weekly Gini comparison across models", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════
#  Sidebar & routing
# ═════════════════════════════════════════════════════════════════

st.sidebar.title("Credit Risk Model Stability")
page = st.sidebar.radio("Navigate to", ["Data Drift", "Model Stability"])

if page == "Data Drift":
    page_data_drift()
else:
    page_model_stability()
