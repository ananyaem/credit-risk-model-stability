"""Kaggle Gini stability metric for the Home Credit competition."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def gini_stability(
    week_num: np.ndarray,
    target: np.ndarray,
    score: np.ndarray,
    w_falling: float = 88.0,
    w_std: float = 0.5,
) -> dict[str, float]:
    """
    Compute the exact Kaggle Gini stability metric.

    For each unique week the Gini coefficient is ``2 * AUC − 1``.  A linear
    regression is fit through the weekly Ginis (indexed 0 … N−1) and the
    final score is::

        mean(gini) + 88.0 * min(0, slope) − 0.5 * std(residuals)

    Parameters
    ----------
    week_num : array-like  – WEEK_NUM per observation.
    target   : array-like  – binary ground truth (0 / 1).
    score    : array-like  – predicted probability of default.
    w_falling: weight on ``min(0, slope)``  (default 88.0).
    w_std    : weight on ``std(residuals)`` (default 0.5, applied as −0.5).

    Returns
    -------
    dict with keys:
        stability_score, mean_gini, falling_rate, std_residuals,
        slope, n_weeks, weekly_ginis
    """
    weeks = np.asarray(week_num)
    y_true = np.asarray(target)
    y_score = np.asarray(score)

    unique_weeks = np.sort(np.unique(weeks))
    weekly_ginis: list[float] = []

    for w in unique_weeks:
        mask = weeks == w
        wk_true = y_true[mask]
        wk_score = y_score[mask]
        if len(np.unique(wk_true)) < 2:
            continue
        weekly_ginis.append(2.0 * roc_auc_score(wk_true, wk_score) - 1.0)

    ginis = np.array(weekly_ginis)
    mean_gini = float(np.mean(ginis)) if len(ginis) else 0.0

    if len(ginis) < 2:
        return {
            "stability_score": mean_gini,
            "mean_gini": mean_gini,
            "falling_rate": 0.0,
            "std_residuals": 0.0,
            "slope": 0.0,
            "n_weeks": len(ginis),
            "weekly_ginis": ginis.tolist(),
        }

    x = np.arange(len(ginis))
    slope, intercept = np.polyfit(x, ginis, 1)
    residuals = ginis - (slope * x + intercept)
    std_residuals = float(np.std(residuals))
    falling_rate = float(min(0.0, slope))

    stability_score = mean_gini + w_falling * falling_rate - w_std * std_residuals

    return {
        "stability_score": float(stability_score),
        "mean_gini": mean_gini,
        "falling_rate": falling_rate,
        "std_residuals": std_residuals,
        "slope": float(slope),
        "n_weeks": len(ginis),
        "weekly_ginis": ginis.tolist(),
    }
