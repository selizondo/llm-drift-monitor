"""
trends.py — Threshold checking and batch summary formatting.

Thresholds are intentionally conservative for the demo — tune for your production system.
PSI > 0.25 and KS > 30% drifted dims are industry standard starting points.
"""

THRESHOLDS = {
    "pct_dims_drifted": 0.30,   # >30% of tested embedding dims drifted
    "psi_query_length": 0.25,   # PSI on query word count
    "centroid_drift": 0.05,     # cosine distance from baseline centroid
    "avg_quality_score": 2.0,   # below 2.0/3.0 (was 1-5 in eval harness; this is 1-3)
    "hallucination_rate": 0.30, # >30% of sampled answers flagged
}


def check_thresholds(metrics: dict) -> dict:
    """Return dict of fired alerts (key → human-readable reason). Empty = all clear."""
    alerts = {}
    if metrics.get("pct_dims_drifted", 0) > THRESHOLDS["pct_dims_drifted"]:
        alerts["embedding_drift"] = (
            f"pct_dims_drifted={metrics['pct_dims_drifted']:.0%} "
            f"(threshold {THRESHOLDS['pct_dims_drifted']:.0%})"
        )
    if metrics.get("psi_query_length", 0) > THRESHOLDS["psi_query_length"]:
        alerts["length_drift"] = (
            f"psi_query_length={metrics['psi_query_length']:.3f} "
            f"(threshold {THRESHOLDS['psi_query_length']})"
        )
    if metrics.get("centroid_drift", 0) > THRESHOLDS["centroid_drift"]:
        alerts["centroid_drift"] = (
            f"centroid_drift={metrics['centroid_drift']:.4f} "
            f"(threshold {THRESHOLDS['centroid_drift']})"
        )
    if 0 < metrics.get("avg_quality_score", 3) < THRESHOLDS["avg_quality_score"]:
        alerts["quality_degraded"] = (
            f"avg_quality_score={metrics['avg_quality_score']:.2f} "
            f"(threshold {THRESHOLDS['avg_quality_score']})"
        )
    if metrics.get("hallucination_rate", 0) > THRESHOLDS["hallucination_rate"]:
        alerts["hallucination_spike"] = (
            f"hallucination_rate={metrics['hallucination_rate']:.0%} "
            f"(threshold {THRESHOLDS['hallucination_rate']:.0%})"
        )
    return alerts


def summarize_batch(batch_num: int, metrics: dict, alerts: dict) -> str:
    status = "ALERT" if alerts else "OK   "
    lines = [f"Batch {batch_num:02d} [{status}]"]
    lines.append(
        f"  drift  : {metrics.get('pct_dims_drifted', 0):.0%} dims | "
        f"PSI={metrics.get('psi_query_length', 0):.3f} | "
        f"centroid={metrics.get('centroid_drift', 0):.4f}"
    )
    lines.append(
        f"  quality: score={metrics.get('avg_quality_score', 0):.2f}/3 | "
        f"halluc={metrics.get('hallucination_rate', 0):.0%}"
    )
    for key, reason in alerts.items():
        lines.append(f"  ! {key}: {reason}")
    return "\n".join(lines)
