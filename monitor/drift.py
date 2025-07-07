"""
drift.py — Statistical drift detection: KS test on embedding dimensions + PSI on query length.

Two signals:
  KS test   — asks whether current batch embeddings come from the same distribution as baseline.
              Run on the 20 highest-variance dimensions (more signal, less noise than all 384).
  PSI       — measures how much query length distribution has shifted.
              PSI < 0.1: stable | 0.1-0.25: watch | > 0.25: alert.

Both signals fire in batches 6-8 (OOD queries are longer and semantically distant from ML Q&A).
"""

import numpy as np
from scipy.stats import ks_2samp


def compute_psi(baseline: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index. Uses baseline bin edges for comparability."""
    baseline_counts, bin_edges = np.histogram(baseline, bins=bins)
    current_counts, _ = np.histogram(current, bins=bin_edges)
    baseline_pct = (baseline_counts + 1e-6) / len(baseline)
    current_pct = (current_counts + 1e-6) / len(current)
    return float(np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct)))


def _select_informative_dims(baseline_embeddings: np.ndarray, n: int = 20) -> list[int]:
    """Pick the n dimensions with highest variance — they carry the most distributional signal."""
    variances = baseline_embeddings.var(axis=0)
    return np.argsort(variances)[-n:].tolist()


def compute_drift_report(
    baseline_embeddings: np.ndarray,
    current_embeddings: np.ndarray,
    baseline_lengths: list[int],
    current_lengths: list[int],
) -> dict:
    """
    Returns a drift report dict with per-dimension KS results, KS on query length,
    PSI on query length (for logging), aggregate counts, and a boolean drift_alert.

    Note: PSI is logged but NOT used for alerting — it's too noisy at n=50 per batch.
    KS test is used for alerting on both embedding dims and query length because
    it accounts for sample size automatically.
    """
    dims = _select_informative_dims(baseline_embeddings)

    ks_results: dict[int, dict] = {}
    for d in dims:
        stat, p_val = ks_2samp(baseline_embeddings[:, d], current_embeddings[:, d])
        ks_results[d] = {
            "ks_stat": round(float(stat), 4),
            "p_value": round(float(p_val), 4),
            "drifted": bool(p_val < 0.05),
        }

    n_drifted = sum(1 for r in ks_results.values() if r["drifted"])
    pct_drifted = n_drifted / len(dims)

    # KS test on query lengths — more reliable than PSI at small sample sizes
    ks_len_stat, ks_len_p = ks_2samp(
        np.array(baseline_lengths, dtype=float),
        np.array(current_lengths, dtype=float),
    )
    # PSI kept for logging; not used as an alert signal
    psi_length = compute_psi(
        np.array(baseline_lengths, dtype=float),
        np.array(current_lengths, dtype=float),
    )

    return {
        "n_dims_tested": len(dims),
        "n_dims_drifted": n_drifted,
        "pct_dims_drifted": round(pct_drifted, 4),
        "ks_length_stat": round(float(ks_len_stat), 4),
        "ks_length_p_value": round(float(ks_len_p), 4),
        "ks_length_drifted": bool(ks_len_p < 0.05),
        "psi_query_length": round(psi_length, 4),
        "drift_alert": pct_drifted > 0.15 or ks_len_p < 0.05,
        "ks_details": {str(k): v for k, v in ks_results.items()},
    }
