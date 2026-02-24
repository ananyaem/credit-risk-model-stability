"""Stacking ensemble with CalibratedClassifierCV(RidgeClassifier) meta-learner.

The meta-learner takes base-model OOF probability predictions as features and
learns calibrated combination weights.  Random seed averaging (default 3 seeds)
trains the calibration pipeline multiple times with different internal CV splits
and averages predictions for robustness.
"""

from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from src.metrics import gini_stability


def _make_meta(alpha: float, seed: int) -> CalibratedClassifierCV:
    """Create a calibrated Ridge meta-learner with a specific CV seed."""
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    return CalibratedClassifierCV(
        RidgeClassifier(alpha=alpha),
        cv=inner_cv,
        method="sigmoid",
    )


def build_stacking_ensemble(
    oof_scores: np.ndarray,
    y: np.ndarray,
    week_num: np.ndarray,
    test_scores: np.ndarray | None = None,
    *,
    seeds: list[int] | None = None,
    alpha: float = 1.0,
    n_splits: int = 5,
    cv_seed: int = 42,
    verbose: bool = True,
) -> dict:
    """Build a stacking ensemble from base-model OOF predictions.

    Parameters
    ----------
    oof_scores  : (n_train, n_models) base-model OOF probability predictions.
    y           : (n_train,) binary target.
    week_num    : (n_train,) WEEK_NUM for StratifiedGroupKFold and stability.
    test_scores : (n_test, n_models) or None — base-model averaged test preds.
    seeds       : Random seeds for seed averaging (default ``[42, 123, 456]``).
    alpha       : Ridge regularisation strength.
    n_splits    : Outer CV folds (should match base-model CV).
    cv_seed     : Random state for the outer StratifiedGroupKFold.
    verbose     : Print per-seed and final metrics.

    Returns
    -------
    dict with keys:
        oof_preds, test_preds, oof_auc, oof_stability, meta_learners
    """
    if seeds is None:
        seeds = [42, 123, 456]

    outer_cv = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=cv_seed,
    )

    oof_ensemble = np.zeros(len(y))
    test_ensemble = (
        np.zeros(len(test_scores)) if test_scores is not None else None
    )
    meta_learners: list[CalibratedClassifierCV] = []

    for seed in seeds:
        if verbose:
            print(f"  Seed {seed}: ", end="", flush=True)

        # --- OOF via outer CV (unbiased meta-predictions) ---
        oof_seed = np.zeros(len(y))
        for train_idx, val_idx in outer_cv.split(oof_scores, y, week_num):
            meta = _make_meta(alpha, seed)
            meta.fit(oof_scores[train_idx], y[train_idx])
            oof_seed[val_idx] = meta.predict_proba(oof_scores[val_idx])[:, 1]

        oof_ensemble += oof_seed / len(seeds)

        seed_auc = roc_auc_score(y, oof_seed)
        seed_stab = gini_stability(week_num, y, oof_seed)
        if verbose:
            print(
                f"AUC={seed_auc:.6f}  "
                f"Stability={seed_stab['stability_score']:.6f}"
            )

        # --- Full-data fit for test predictions ---
        meta_full = _make_meta(alpha, seed)
        meta_full.fit(oof_scores, y)
        meta_learners.append(meta_full)

        if test_scores is not None:
            test_ensemble += (
                meta_full.predict_proba(test_scores)[:, 1] / len(seeds)
            )

    # --- Evaluate the seed-averaged ensemble ---
    oof_auc = roc_auc_score(y, oof_ensemble)
    oof_stab = gini_stability(week_num, y, oof_ensemble)

    if verbose:
        print(f"\n  Ensemble ({len(seeds)}-seed avg):")
        print(f"    AUC:            {oof_auc:.6f}")
        print(f"    Stability:      {oof_stab['stability_score']:.6f}")
        print(f"    Mean Gini:      {oof_stab['mean_gini']:.6f}")
        print(f"    Falling rate:   {oof_stab['falling_rate']:.6f}")
        print(f"    Std residuals:  {oof_stab['std_residuals']:.6f}")

    return {
        "oof_preds": oof_ensemble,
        "test_preds": test_ensemble,
        "oof_auc": oof_auc,
        "oof_stability": oof_stab,
        "meta_learners": meta_learners,
    }
