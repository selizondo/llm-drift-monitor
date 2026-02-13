"""
trends.py — Threshold checking and batch summary formatting.

Thresholds are intentionally conservative for the demo — tune for your production system.
PSI > 0.25 and KS > 30% drifted dims are industry standard starting points.
"""

import os

THRESHOLDS = {
    # >15% of tested embedding dims drifted (KS p<0.05 per dim)
    # Calibrated from observed data: in-distribution batches score 0%, OOD batches 15-25%
    "pct_dims_drifted": float(os.getenv("THRESHOLD_PCT_DIMS_DRIFTED", "0.15")),
    # KS test on query word count — p<0.05 means length distribution shifted
    # More reliable than PSI at n=50 per batch (PSI is designed for large samples)
    "ks_length_p_value": float(os.getenv("THRESHOLD_KS_LENGTH_P_VALUE", "0.05")),
    # Cosine distance from baseline embedding centroid
    # In-distribution variance: 0.00–0.19; OOD batches: 0.54–0.60
    # Threshold 0.30 leaves clear margin on both sides
    "centroid_drift": float(os.getenv("THRESHOLD_CENTROID_DRIFT", "0.30")),
    # Avg cosine sim between query and best match in ML corpus
    # In-distribution: ~0.5-0.7; OOD: ~0.1-0.3. Threshold 0.35 catches OOD.
    "avg_retrieval_sim": float(os.getenv("THRESHOLD_AVG_RETRIEVAL_SIM", "0.35")),
    # LLM judge score — only meaningful with larger sample sizes (n≥15)
    # At n=5, each flag shifts rate by 20%; use for qualitative trends, not hard alerts
    "avg_quality_score": float(os.getenv("THRESHOLD_AVG_QUALITY_SCORE", "2.0")),
    "hallucination_rate": float(os.getenv("THRESHOLD_HALLUCINATION_RATE", "0.30")),
}


def check_thresholds(metrics: dict, llm_judge_enabled: bool = False) -> dict:
    """
    Return dict of fired alerts (key → human-readable reason). Empty = all clear.

    LLM judge alert keys (quality_degraded, hallucination_spike) are only generated when:
      - llm_judge_enabled=True (caller ran score_batch_sample with an API client), AND
      - metrics["n_sampled"] >= 15 (sufficient sample for statistical reliability)
    At n=5, each hallucination flag shifts the rate by 20% — too noisy for hard alerts.
    """
    alerts = {}
    if metrics.get("pct_dims_drifted", 0) > THRESHOLDS["pct_dims_drifted"]:
        alerts["embedding_drift"] = (
            f"pct_dims_drifted={metrics['pct_dims_drifted']:.0%} "
            f"(threshold {THRESHOLDS['pct_dims_drifted']:.0%})"
        )
    if metrics.get("ks_length_p_value", 1.0) < THRESHOLDS["ks_length_p_value"]:
        alerts["length_drift"] = (
            f"ks_length_p={metrics['ks_length_p_value']:.4f} "
            f"(KS test: query length distribution shifted, p<{THRESHOLDS['ks_length_p_value']})"
        )
    if metrics.get("centroid_drift", 0) > THRESHOLDS["centroid_drift"]:
        alerts["centroid_drift"] = (
            f"centroid_drift={metrics['centroid_drift']:.4f} "
            f"(threshold {THRESHOLDS['centroid_drift']})"
        )
    if 0 < metrics.get("avg_retrieval_sim", 1.0) < THRESHOLDS["avg_retrieval_sim"]:
        alerts["retrieval_degraded"] = (
            f"avg_retrieval_sim={metrics['avg_retrieval_sim']:.3f} "
            f"(threshold {THRESHOLDS['avg_retrieval_sim']})"
        )
    # LLM judge alerts: only fire when caller explicitly enables them AND n≥15.
    # n_sampled < 15: each flag shifts rate by >6.7% — false positive risk too high.
    if llm_judge_enabled and metrics.get("n_sampled", 0) >= 15:
        if metrics.get("avg_quality_score", 3.0) < THRESHOLDS["avg_quality_score"]:
            alerts["quality_degraded"] = (
                f"avg_quality_score={metrics['avg_quality_score']:.2f} "
                f"(threshold {THRESHOLDS['avg_quality_score']}, n={metrics['n_sampled']})"
            )
        if metrics.get("hallucination_rate", 0) > THRESHOLDS["hallucination_rate"]:
            alerts["hallucination_spike"] = (
                f"hallucination_rate={metrics['hallucination_rate']:.0%} "
                f"(threshold {THRESHOLDS['hallucination_rate']:.0%}, n={metrics['n_sampled']})"
            )
    return alerts


def summarize_batch(batch_num: int, metrics: dict, alerts: dict) -> str:
    status = "ALERT" if alerts else "OK   "
    lines = [f"Batch {batch_num:02d} [{status}]"]
    lines.append(
        f"  drift  : {metrics.get('pct_dims_drifted', 0):.0%} dims | "
        f"len_p={metrics.get('ks_length_p_value', 1.0):.3f} | "
        f"centroid={metrics.get('centroid_drift', 0):.4f}"
    )
    lines.append(
        f"  quality: retrieval_sim={metrics.get('avg_retrieval_sim', 0):.3f} | "
        f"miss={metrics.get('retrieval_miss_rate', 0):.0%}"
        + (
            f" | llm_score={metrics.get('avg_quality_score', 0):.2f}/3"
            if metrics.get("avg_quality_score", 0) > 0
            else ""
        )
    )
    for key, reason in alerts.items():
        lines.append(f"  ! {key}: {reason}")
    return "\n".join(lines)
