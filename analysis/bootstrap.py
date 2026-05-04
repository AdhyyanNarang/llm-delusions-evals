"""Bootstrap confidence interval utilities for binary proportions.

Provides functions to compute bootstrapped 95% CIs at per-code,
per-category, and overall levels, respecting the nested structure
of the evaluation data (samples within windows within codes).
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import bootstrap


@dataclass
class BootstrapConfig:
    """Configuration for bootstrap confidence interval calculations.

    Attributes
    ----------
    n_boot:
        Number of bootstrap resamples.
    ci:
        Confidence level (default 0.95 for 95% CI).
    seed:
        Random seed for reproducibility.
    """

    n_boot: int = 10_000
    ci: float = 0.95
    seed: Optional[int] = 42


def bootstrap_binary_ci(
    values: np.ndarray,
    *,
    config: Optional[BootstrapConfig] = None,
) -> tuple[float, float, float]:
    """Compute a bootstrapped confidence interval for a binary proportion.

    Parameters
    ----------
    values:
        1-D array of binary (0/1) values.
    config:
        Bootstrap configuration. If None, uses default settings.

    Returns
    -------
    tuple[float, float, float]
        ``(mean, ci_lower, ci_upper)`` as proportions in [0, 1].
    """
    if config is None:
        config = BootstrapConfig()

    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return (np.nan, np.nan, np.nan)

    observed_mean = float(np.mean(values))
    if len(values) == 1:
        return (observed_mean, observed_mean, observed_mean)

    rng = np.random.default_rng(config.seed)
    result = bootstrap(
        (values,),
        np.mean,
        n_resamples=config.n_boot,
        confidence_level=config.ci,
        method="percentile",
        vectorized=True,
        rng=rng,
    )

    return (
        observed_mean,
        float(result.confidence_interval.low),
        float(result.confidence_interval.high),
    )


def bootstrap_grouped_ci(
    df: pd.DataFrame,
    *,
    group_col: str,
    score_col: str = "score",
    config: Optional[BootstrapConfig] = None,
) -> pd.DataFrame:
    """Compute bootstrapped CIs grouped by a column.

    Parameters
    ----------
    df:
        Input DataFrame containing at least ``group_col`` and ``score_col``.
    group_col:
        Column to group by (e.g. ``"code_short"`` or ``"category"``).
    score_col:
        Column containing binary 0/1 scores.
    config:
        Bootstrap configuration. If None, uses default settings.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ``group_col``, ``mean``, ``ci_lower``,
        ``ci_upper``, ``n``.
    """
    if config is None:
        config = BootstrapConfig()

    results = []
    for group_value, group_df in df.groupby(group_col):
        values = group_df[score_col].dropna().values
        mean_val, lower, upper = bootstrap_binary_ci(values, config=config)
        results.append(
            {
                group_col: group_value,
                "mean": mean_val,
                "ci_lower": lower,
                "ci_upper": upper,
                "n": len(values),
            }
        )
    return pd.DataFrame(results)


def bootstrap_delta_ci(
    values_a: np.ndarray,
    values_b: np.ndarray,
    *,
    config: Optional[BootstrapConfig] = None,
) -> tuple[float, float, float]:
    """Compute a bootstrapped CI for the difference in means (a - b).

    Resamples the two groups independently on each iteration, then
    computes the difference of their means.

    Parameters
    ----------
    values_a:
        1-D array of binary (0/1) values for group A.
    values_b:
        1-D array of binary (0/1) values for group B.
    config:
        Bootstrap configuration. If None, uses default settings.

    Returns
    -------
    tuple[float, float, float]
        ``(observed_delta, ci_lower, ci_upper)`` as proportions.
    """
    if config is None:
        config = BootstrapConfig()

    values_a = np.asarray(values_a, dtype=float)
    values_b = np.asarray(values_b, dtype=float)
    values_a = values_a[~np.isnan(values_a)]
    values_b = values_b[~np.isnan(values_b)]

    if len(values_a) == 0 or len(values_b) == 0:
        return (np.nan, np.nan, np.nan)

    observed_delta = float(np.mean(values_a) - np.mean(values_b))
    if len(values_a) == 1 and len(values_b) == 1:
        return (observed_delta, observed_delta, observed_delta)

    rng = np.random.default_rng(config.seed)
    result = bootstrap(
        (values_a, values_b),
        _mean_delta_statistic,
        n_resamples=config.n_boot,
        confidence_level=config.ci,
        method="percentile",
        vectorized=True,
        paired=False,
        rng=rng,
    )

    return (
        observed_delta,
        float(result.confidence_interval.low),
        float(result.confidence_interval.high),
    )


def _mean_delta_statistic(
    values_a: np.ndarray, values_b: np.ndarray, axis: int
) -> np.ndarray:
    """Return mean(values_a) - mean(values_b) along ``axis``."""
    return np.mean(values_a, axis=axis) - np.mean(values_b, axis=axis)


def bootstrap_model_code_ci(
    df: pd.DataFrame,
    *,
    model_col: str = "model_label",
    code_col: str = "code_short",
    score_col: str = "score",
    config: Optional[BootstrapConfig] = None,
) -> pd.DataFrame:
    """Compute bootstrapped CIs for each (model, code) combination.

    Parameters
    ----------
    df:
        Input DataFrame.
    model_col:
        Column identifying the model.
    code_col:
        Column identifying the annotation code.
    score_col:
        Column containing binary 0/1 scores.
    config:
        Bootstrap configuration. If None, uses default settings.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ``model_col``, ``code_col``, ``mean``,
        ``ci_lower``, ``ci_upper``, ``n``.
    """
    if config is None:
        config = BootstrapConfig()

    results = []
    for (model, code), group_df in df.groupby([model_col, code_col]):
        values = group_df[score_col].dropna().values
        mean_val, lower, upper = bootstrap_binary_ci(values, config=config)
        results.append(
            {
                model_col: model,
                code_col: code,
                "mean": mean_val,
                "ci_lower": lower,
                "ci_upper": upper,
                "n": len(values),
            }
        )
    return pd.DataFrame(results)
