"""Statistical analysis of reasoning-effort effects within model families.

For each model family with multiple reasoning settings (e.g., GPT-5.4 default
vs. high), computes per-code prevalence deltas with bootstrapped 95% CIs and
flags whether the CI excludes zero. Also splits scores by turn-bin (early /
mid / late thirds within each window) to see where the effect concentrates.

Outputs CSVs under ``analysis/data/reasoning_effects/`` so the LaTeX text can
cite specific deltas with significance markers.

Usage::

    python -m analysis.compute_reasoning_effects
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from analysis.bootstrap import BootstrapConfig, bootstrap_delta_ci
from analysis.load_eval_data import load_all_eval_data

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent / "data" / "reasoning_effects"

# Model-family comparisons: (family_name, baseline_label, contrast_labels).
FAMILIES: list[tuple[str, str, list[str]]] = [
    ("gpt54", "GPT-5.4", ["GPT-5.4 (high)"]),
    # Qwen3.5-397B has no default-reasoning run, so we use (low) as the
    # baseline and contrast against (high).
    ("qwen397b", "Qwen3.5-397B (low)", ["Qwen3.5-397B (high)"]),
]

N_TURN_BINS = 3
TURN_BIN_LABELS = ["early", "mid", "late"]
MIN_VARIANTS_FOR_COMPARISON = 2


def _assign_turn_bin(df: pd.DataFrame) -> pd.DataFrame:
    """Assign each row to an early/mid/late bin within its window.

    Bins are computed per ``window_id`` using equal-width quantile cuts on
    ``turn_index`` so that windows of different lengths are comparable.
    """
    out = df.copy()
    out["turn_bin"] = pd.NA

    valid = out["turn_index"].notna()
    sub = out.loc[valid].copy()
    if sub.empty:
        return out

    def _bin_window(group: pd.DataFrame) -> pd.Series:
        ranks = group["turn_index"].rank(method="first")
        # qcut may fail for windows shorter than N_TURN_BINS; fall back to
        # placing all turns in the same bin.
        try:
            return pd.qcut(
                ranks, N_TURN_BINS, labels=TURN_BIN_LABELS, duplicates="drop"
            )
        except ValueError:
            return pd.Series([TURN_BIN_LABELS[0]] * len(group), index=group.index)

    binned = sub.groupby("window_id", group_keys=False).apply(_bin_window)
    out.loc[binned.index, "turn_bin"] = binned.astype(str)
    return out


def _delta_row(
    group_label: str,
    labels: tuple[str, str],
    values: tuple[np.ndarray, np.ndarray],
    config: BootstrapConfig,
) -> dict:
    """Compute one delta row (contrast - baseline) with bootstrap CI.

    Parameters
    ----------
    group_label:
        Category or code key being compared.
    labels:
        ``(baseline_label, contrast_label)`` tuple identifying the two arms.
    values:
        ``(baseline_vals, contrast_vals)`` arrays of binary 0/1 scores.
    config:
        Bootstrap configuration.
    """
    baseline_label, contrast_label = labels
    baseline_vals, contrast_vals = values
    delta, lo, hi = bootstrap_delta_ci(contrast_vals, baseline_vals, config=config)
    significant = not np.isnan(lo) and not np.isnan(hi) and (lo > 0 or hi < 0)
    return {
        "group": group_label,
        "baseline": baseline_label,
        "contrast": contrast_label,
        "n_baseline": int((~np.isnan(baseline_vals)).sum()),
        "n_contrast": int((~np.isnan(contrast_vals)).sum()),
        "baseline_mean": float(np.nanmean(baseline_vals))
        if len(baseline_vals)
        else np.nan,
        "contrast_mean": float(np.nanmean(contrast_vals))
        if len(contrast_vals)
        else np.nan,
        "delta": delta,
        "ci_lower": lo,
        "ci_upper": hi,
        "significant_95ci": significant,
    }


def _compute_family_deltas(
    df: pd.DataFrame,
    comparison: tuple[str, list[str]],
    *,
    grouping: str,
    config: BootstrapConfig,
    turn_bin: Optional[str] = None,
) -> pd.DataFrame:
    """Compute deltas across one grouping (category or code_short).

    Parameters
    ----------
    df:
        Tidy row-level DataFrame with ``score``, ``model_label``, and the
        ``grouping`` column.
    comparison:
        ``(baseline_label, contrast_labels)`` pair selecting the model arms.
    grouping:
        Column to group by (e.g., ``"category"`` or ``"code_short"``).
    config:
        Bootstrap configuration.
    turn_bin:
        Optional turn-bin filter; when set, only rows in that bin are used.

    Each output row reports (contrast - baseline) prevalence with a 95% CI.
    """
    baseline_label, contrast_labels = comparison
    rows: list[dict] = []
    sub = df if turn_bin is None else df[df["turn_bin"] == turn_bin]

    baseline_df = sub[sub["model_label"] == baseline_label]
    if baseline_df.empty:
        logger.warning(
            "No baseline rows for %s (turn_bin=%s)", baseline_label, turn_bin
        )
        return pd.DataFrame(rows)

    groups = sorted(g for g in sub[grouping].dropna().unique())

    for contrast_label in contrast_labels:
        contrast_df = sub[sub["model_label"] == contrast_label]
        if contrast_df.empty:
            continue
        for grp in groups:
            b_vals = baseline_df.loc[baseline_df[grouping] == grp, "score"].to_numpy(
                dtype=float
            )
            c_vals = contrast_df.loc[contrast_df[grouping] == grp, "score"].to_numpy(
                dtype=float
            )
            if len(b_vals) == 0 or len(c_vals) == 0:
                continue
            row = _delta_row(
                grp, (baseline_label, contrast_label), (b_vals, c_vals), config
            )
            if turn_bin is not None:
                row["turn_bin"] = turn_bin
            rows.append(row)

    return pd.DataFrame(rows)


def compute_all() -> dict[str, pd.DataFrame]:
    """Compute reasoning-effect deltas for every configured family.

    Returns a dict with keys per family containing category-level, code-level,
    and turn-bin-split code-level delta tables.
    """
    df = load_all_eval_data(include_original_transcript=False)
    df = df[df["score"].notna()].copy()
    df = _assign_turn_bin(df)

    config = BootstrapConfig()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results: dict[str, pd.DataFrame] = {}

    for family_name, baseline, contrasts in FAMILIES:
        present = [baseline] + [c for c in contrasts if c in df["model_label"].unique()]
        if (
            baseline not in df["model_label"].unique()
            or len(present) < MIN_VARIANTS_FOR_COMPARISON
        ):
            logger.warning("Skipping %s: variants not present in data", family_name)
            continue

        comparison = (baseline, contrasts)
        cat_deltas = _compute_family_deltas(
            df, comparison, grouping="category", config=config
        )
        code_deltas = _compute_family_deltas(
            df, comparison, grouping="code_short", config=config
        )

        bin_rows = []
        for tb in TURN_BIN_LABELS:
            bin_rows.append(
                _compute_family_deltas(
                    df,
                    comparison,
                    grouping="code_short",
                    config=config,
                    turn_bin=tb,
                )
            )
        bin_deltas = (
            pd.concat(bin_rows, ignore_index=True) if bin_rows else pd.DataFrame()
        )

        cat_path = OUTPUT_DIR / f"{family_name}_category_deltas.csv"
        code_path = OUTPUT_DIR / f"{family_name}_code_deltas.csv"
        bin_path = OUTPUT_DIR / f"{family_name}_code_deltas_by_turn_bin.csv"
        cat_deltas.to_csv(cat_path, index=False)
        code_deltas.to_csv(code_path, index=False)
        bin_deltas.to_csv(bin_path, index=False)
        logger.info("Wrote %s, %s, %s", cat_path, code_path, bin_path)

        results[family_name] = cat_deltas
        results[f"{family_name}_codes"] = code_deltas
        results[f"{family_name}_codes_by_turn_bin"] = bin_deltas

    return results


def _format_pp(value: float) -> str:
    """Format a proportion as a signed percentage-point string."""
    if np.isnan(value):
        return "n/a"
    return f"{value * 100:+.1f}pp"


def print_summary(results: dict[str, pd.DataFrame]) -> None:
    """Print a human-readable summary of significant deltas."""
    for key, df_ in results.items():
        if df_.empty:
            continue
        print(f"\n=== {key} ===")
        for _, row in df_.iterrows():
            marker = "*" if row.get("significant_95ci") else " "
            tb = (
                f" [{row['turn_bin']}]"
                if "turn_bin" in row and pd.notna(row.get("turn_bin"))
                else ""
            )
            ci_lo = _format_pp(row["ci_lower"])
            ci_hi = _format_pp(row["ci_upper"])
            print(
                f"  {marker} {row['contrast']} vs {row['baseline']} "
                f"| {row['group']}{tb}: "
                f"{_format_pp(row['delta'])} "
                f"(95% CI [{ci_lo}, {ci_hi}])"
            )


def main() -> None:
    """Entry point for command-line execution."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    results = compute_all()
    print_summary(results)


if __name__ == "__main__":
    main()
