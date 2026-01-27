"""
dashboard.py — W&B logging: time-series metrics, alert flags, live dashboard.

W&B owns the "what is happening now" question:
  - All metrics logged with batch number as the x-axis step.
  - Drift signals and quality signals in separate namespaces (drift/ quality/ alerts/).
  - Alert flags are logged as 0/1 integers so W&B can draw threshold lines.

See docs/adr-01-tool-split.md for the MLflow vs W&B ownership boundary.
"""

import wandb

PROJECT_NAME = "llm-drift-monitor"
_run = None


def init_run(run_name: str = "monitoring-session") -> None:
    global _run
    _run = wandb.init(project=PROJECT_NAME, name=run_name, reinit=True)
    wandb.define_metric("batch")
    wandb.define_metric("*", step_metric="batch")


def log_batch(batch_num: int, metrics: dict, alerts: dict) -> None:
    payload = {"batch": batch_num}
    payload.update({
        "drift/pct_dims_drifted": metrics.get("pct_dims_drifted", 0),
        "drift/psi_query_length": metrics.get("psi_query_length", 0),
        "drift/centroid_cosine": metrics.get("centroid_drift", 0),
        "quality/avg_retrieval_sim": metrics.get("avg_retrieval_sim", 0),
        "quality/retrieval_miss_rate": metrics.get("retrieval_miss_rate", 0),
        "quality/llm_score": metrics.get("avg_quality_score", 0),
        "quality/hallucination_rate": metrics.get("hallucination_rate", 0),
        "alerts/total": len(alerts),
        "alerts/embedding_drift": int("embedding_drift" in alerts),
        "alerts/length_drift": int("length_drift" in alerts),
        "alerts/centroid_drift": int("centroid_drift" in alerts),
        # quality_degraded / hallucination_spike are only present when check_thresholds()
        # is called with llm_judge_enabled=True and n_sampled >= 15. Log as 0 otherwise
        # so the W&B time series stays continuous (no gaps in the alert lines).
        "alerts/quality_degraded": int("quality_degraded" in alerts),
        "alerts/hallucination_spike": int("hallucination_spike" in alerts),
    })
    wandb.log(payload)


def finish() -> None:
    if _run is not None:
        wandb.finish()
