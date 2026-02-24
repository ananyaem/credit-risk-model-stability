"""Stability-aware hyperparameter tuning with Optuna.

Each tuning function runs StratifiedGroupKFold cross-validation where the
objective is the **Gini stability metric** (not raw AUC).  This aligns the
hyper-parameter search directly with the competition's evaluation criterion.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.metrics import gini_stability

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ──────────────────────────────────────────────────────────────
#  CatBoost
# ──────────────────────────────────────────────────────────────

def tune_catboost(
    X: pd.DataFrame,
    y: np.ndarray,
    week_num: np.ndarray,
    cat_cols: list[str],
    *,
    n_trials: int = 50,
    n_splits: int = 5,
    seed: int = 42,
) -> dict:
    """Tune CatBoost hyper-parameters by maximising Gini stability.

    Parameters
    ----------
    X         : Feature DataFrame (all columns, including categoricals).
    y         : Binary target array.
    week_num  : WEEK_NUM array used for StratifiedGroupKFold grouping.
    cat_cols  : Names of columns treated as native categoricals by CatBoost.
    n_trials  : Number of Optuna trials.
    n_splits  : Number of CV folds.
    seed      : Random seed for CV and models.
    """
    from catboost import CatBoostClassifier

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    def objective(trial: optuna.Trial) -> float:
        params = dict(
            iterations=1000,
            depth=trial.suggest_int("depth", 3, 8),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 0.1, 30.0, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bylevel=trial.suggest_float("colsample_bylevel", 0.5, 1.0),
            bootstrap_type="Bernoulli",
            random_seed=seed,
            eval_metric="AUC",
            cat_features=cat_cols,
            allow_writing_files=False,
            verbose=0,
        )

        oof = np.zeros(len(y))
        for train_idx, val_idx in sgkf.split(X, y, week_num):
            model = CatBoostClassifier(**params)
            model.fit(
                X.iloc[train_idx], y[train_idx],
                eval_set=(X.iloc[val_idx], y[val_idx]),
                early_stopping_rounds=50,
                verbose=0,
            )
            oof[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]

        return gini_stability(week_num, y, oof)["stability_score"]

    study = optuna.create_study(direction="maximize", study_name="catboost_stability")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    best["bootstrap_type"] = "Bernoulli"
    return {
        "best_params": best,
        "best_stability": study.best_value,
        "n_trials": len(study.trials),
    }


# ──────────────────────────────────────────────────────────────
#  LightGBM
# ──────────────────────────────────────────────────────────────

def tune_lightgbm(
    X: pd.DataFrame,
    y: np.ndarray,
    week_num: np.ndarray,
    cat_cols: list[str],
    *,
    n_trials: int = 50,
    n_splits: int = 5,
    seed: int = 42,
) -> dict:
    """Tune LightGBM hyper-parameters by maximising Gini stability.

    Categoricals are converted to ``category`` dtype for native LightGBM
    support.  The caller should already have excluded high-cardinality
    categoricals (>200 unique values) from *X* and *cat_cols*.
    """
    import lightgbm as lgb

    X_lgb = X.copy()
    for col in cat_cols:
        if col in X_lgb.columns:
            X_lgb[col] = X_lgb[col].astype("category")

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    def objective(trial: optuna.Trial) -> float:
        params = dict(
            n_estimators=1000,
            max_depth=trial.suggest_int("max_depth", -1, 12),
            num_leaves=trial.suggest_int("num_leaves", 16, 255),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 0.1, 30.0, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            random_state=seed,
            verbose=-1,
        )

        oof = np.zeros(len(y))
        for train_idx, val_idx in sgkf.split(X_lgb, y, week_num):
            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_lgb.iloc[train_idx], y[train_idx],
                eval_set=[(X_lgb.iloc[val_idx], y[val_idx])],
                eval_metric="auc",
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
            oof[val_idx] = model.predict_proba(X_lgb.iloc[val_idx])[:, 1]

        return gini_stability(week_num, y, oof)["stability_score"]

    study = optuna.create_study(direction="maximize", study_name="lightgbm_stability")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    return {
        "best_params": study.best_params,
        "best_stability": study.best_value,
        "n_trials": len(study.trials),
    }


# ──────────────────────────────────────────────────────────────
#  XGBoost  (fold-safe CatBoost encoding for categoricals)
# ──────────────────────────────────────────────────────────────

def tune_xgboost(
    X: pd.DataFrame,
    y: np.ndarray,
    week_num: np.ndarray,
    cat_cols: list[str],
    *,
    n_trials: int = 50,
    n_splits: int = 5,
    seed: int = 42,
) -> dict:
    """Tune XGBoost hyper-parameters by maximising Gini stability.

    Categorical features are encoded **per-fold** with ``CatBoostEncoder``
    (strictly fold-safe: encoder fitted only on each fold's training split).
    The caller should already have excluded high-cardinality categoricals
    (>200 unique values) from *X* and *cat_cols*.
    """
    from category_encoders import CatBoostEncoder
    from xgboost import XGBClassifier

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    def objective(trial: optuna.Trial) -> float:
        params = dict(
            n_estimators=1000,
            max_depth=trial.suggest_int("max_depth", 3, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 0.1, 30.0, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            random_state=seed,
            eval_metric="auc",
            tree_method="hist",
            early_stopping_rounds=50,
            verbosity=0,
        )

        oof = np.zeros(len(y))
        for fold_i, (train_idx, val_idx) in enumerate(sgkf.split(X, y, week_num)):
            X_tr = X.iloc[train_idx].copy()
            X_val = X.iloc[val_idx].copy()

            if cat_cols:
                enc = CatBoostEncoder(cols=cat_cols, random_state=seed + fold_i)
                X_tr[cat_cols] = enc.fit_transform(X_tr[cat_cols], y[train_idx])
                X_val[cat_cols] = enc.transform(X_val[cat_cols])

            model = XGBClassifier(**params)
            model.fit(X_tr, y[train_idx], eval_set=[(X_val, y[val_idx])], verbose=0)
            oof[val_idx] = model.predict_proba(X_val)[:, 1]

        return gini_stability(week_num, y, oof)["stability_score"]

    study = optuna.create_study(direction="maximize", study_name="xgboost_stability")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    return {
        "best_params": study.best_params,
        "best_stability": study.best_value,
        "n_trials": len(study.trials),
    }


# ──────────────────────────────────────────────────────────────
#  Run all three and persist
# ──────────────────────────────────────────────────────────────

def tune_all(
    X_full: pd.DataFrame,
    y: np.ndarray,
    week_num: np.ndarray,
    cat_cols_full: list[str],
    X_reduced: pd.DataFrame,
    cat_cols_reduced: list[str],
    *,
    n_trials: int = 50,
    n_splits: int = 5,
    seed: int = 42,
    output_path: str | Path = "artifacts/best_params.json",
) -> dict:
    """Run Optuna studies for all three models and persist best params.

    Parameters
    ----------
    X_full           : Feature DataFrame for CatBoost (all features).
    cat_cols_full    : Categorical column names for CatBoost.
    X_reduced        : Feature DataFrame for LightGBM / XGBoost
                       (high-cardinality categoricals removed).
    cat_cols_reduced : Categorical column names for LightGBM / XGBoost.
    output_path      : Where to write the JSON with best params.
    """
    results: dict[str, dict] = {}

    print("═" * 60)
    print("  Tuning CatBoost")
    print("═" * 60)
    results["catboost"] = tune_catboost(
        X_full, y, week_num, cat_cols_full,
        n_trials=n_trials, n_splits=n_splits, seed=seed,
    )
    print(f"  Best stability: {results['catboost']['best_stability']:.6f}")
    print(f"  Best params:    {results['catboost']['best_params']}")

    print(f"\n{'═' * 60}")
    print("  Tuning LightGBM")
    print("═" * 60)
    results["lightgbm"] = tune_lightgbm(
        X_reduced, y, week_num, cat_cols_reduced,
        n_trials=n_trials, n_splits=n_splits, seed=seed,
    )
    print(f"  Best stability: {results['lightgbm']['best_stability']:.6f}")
    print(f"  Best params:    {results['lightgbm']['best_params']}")

    print(f"\n{'═' * 60}")
    print("  Tuning XGBoost")
    print("═" * 60)
    results["xgboost"] = tune_xgboost(
        X_reduced, y, week_num, cat_cols_reduced,
        n_trials=n_trials, n_splits=n_splits, seed=seed,
    )
    print(f"  Best stability: {results['xgboost']['best_stability']:.6f}")
    print(f"  Best params:    {results['xgboost']['best_params']}")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nBest params persisted to {out}")

    return results
