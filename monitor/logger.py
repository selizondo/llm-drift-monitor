"""
logger.py — MLflow logging: per-batch audit trail, baseline artifact, drift reports.

MLflow owns the "what happened and when" question:
  - Each batch is a separate run under the llm-drift-monitor experiment.
  - Baseline stats saved as artifact in a dedicated baseline run.
  - Drift report JSON logged per batch for post-hoc investigation.
  - Tags link batches to alert events for traceability.

See docs/adr-01-tool-split.md for the MLflow vs W&B ownership boundary.
"""

import mlflow

EXPERIMENT_NAME = "llm-drift-monitor"
_initialized = False


def init_experiment() -> None:
    global _initialized
    mlflow.set_experiment(EXPERIMENT_NAME)
    _initialized = True


def log_baseline(baseline_stats: dict) -> str:
    with mlflow.start_run(run_name="baseline"):
        mlflow.log_params({
            "n_samples": baseline_stats["n_samples"],
            "embedding_model": baseline_stats.get("embedding_model", "all-MiniLM-L6-v2"),
        })
        mlflow.log_dict(baseline_stats, "baseline_stats.json")
        run_id = mlflow.active_run().info.run_id
    return run_id


def log_batch(batch_num: int, metrics: dict, alerts: dict, drift_report: dict) -> str:
    with mlflow.start_run(run_name=f"batch-{batch_num:02d}"):
        mlflow.log_metrics({
            "pct_dims_drifted": metrics.get("pct_dims_drifted", 0),
            "psi_query_length": metrics.get("psi_query_length", 0),
            "centroid_drift": metrics.get("centroid_drift", 0),
            "avg_retrieval_sim": metrics.get("avg_retrieval_sim", 0),
            "retrieval_miss_rate": metrics.get("retrieval_miss_rate", 0),
            "avg_quality_score": metrics.get("avg_quality_score", 0),
            "hallucination_rate": metrics.get("hallucination_rate", 0),
            "n_alerts": len(alerts),
        })
        mlflow.set_tags({
            "batch": str(batch_num),
            "drift_alert": str(len(alerts) > 0).lower(),
            "alert_types": ",".join(sorted(alerts.keys())) if alerts else "none",
        })
        mlflow.log_dict(drift_report, "drift_report.json")
        if alerts:
            mlflow.log_dict(alerts, "alerts.json")
        run_id = mlflow.active_run().info.run_id
    return run_id
