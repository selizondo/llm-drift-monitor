#!/usr/bin/env python3
"""Generate a dashboard specification for the LLM drift monitoring service.

Outputs a structured dashboard definition (JSON + Markdown) covering the
four golden signals: latency, traffic, errors, saturation — applied to
the LLM monitoring context.

No external dependencies — stdlib only.

Usage:
    python scripts/dashboard_generator.py                        # stdout Markdown
    python scripts/dashboard_generator.py --format json         # JSON spec
    python scripts/dashboard_generator.py --export dashboard.md # write to file
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


DASHBOARD_SPEC = {
    "name": "LLM Drift Monitor — Operations Dashboard",
    "generated_at": datetime.now().strftime("%Y-%m-%d"),
    "description": "Four golden signals applied to the LLM drift monitoring pipeline. "
                   "Panels are ordered: leading indicators first, lagging second, operational third.",
    "sections": [
        {
            "title": "Leading Indicators — Embedding Drift",
            "signal": "traffic + errors",
            "rationale": "Fire 2–3 batches before retrieval quality degrades. Check here first.",
            "panels": [
                {
                    "name": "KS Statistic — Top-20 Embedding Dimensions",
                    "type": "time_series",
                    "metric": "ks_stat_max",
                    "alert_threshold": 0.30,
                    "description": "Max KS statistic across top-20 variance dims per batch. "
                                   "Threshold 0.30 calibrated empirically from 50-batch baseline.",
                    "y_axis": {"min": 0.0, "max": 1.0, "label": "KS statistic"},
                    "reference_lines": [
                        {"value": 0.30, "label": "Alert threshold", "color": "red"},
                        {"value": 0.10, "label": "Normal ceiling (observed)", "color": "green"},
                    ],
                },
                {
                    "name": "Centroid Cosine Distance",
                    "type": "time_series",
                    "metric": "centroid_dist",
                    "alert_threshold": 0.30,
                    "description": "Cosine distance from batch centroid to baseline centroid. "
                                   "Detects directional drift; complements KS spread detection.",
                    "y_axis": {"min": 0.0, "max": 1.0, "label": "Cosine distance"},
                    "reference_lines": [
                        {"value": 0.30, "label": "Alert threshold", "color": "red"},
                        {"value": 0.19, "label": "In-distribution max (observed)", "color": "green"},
                    ],
                },
                {
                    "name": "PSI — Distribution Shift Trend",
                    "type": "time_series",
                    "metric": "psi_score",
                    "alert_threshold": None,
                    "description": "Population Stability Index — visualization only (not used for alerts "
                                   "at n=50; PSI requires n≥1000 for reliable calibration). "
                                   "Trend direction is informative even if absolute value is noisy.",
                    "y_axis": {"min": 0.0, "max": 0.5, "label": "PSI"},
                    "reference_lines": [
                        {"value": 0.10, "label": "Minor shift (industry convention)", "color": "yellow"},
                        {"value": 0.25, "label": "Major shift (industry convention)", "color": "red"},
                    ],
                },
            ],
        },
        {
            "title": "Lagging Indicators — Retrieval Quality",
            "signal": "errors",
            "rationale": "Confirm embedding drift is causing quality degradation. Check after leading indicators fire.",
            "panels": [
                {
                    "name": "Mean Retrieval Similarity",
                    "type": "time_series",
                    "metric": "mean_retrieval_sim",
                    "alert_threshold": 0.35,
                    "description": "Mean cosine similarity between query embeddings and retrieved chunks. "
                                   "Drops after embedding drift; use to confirm drift is impacting quality.",
                    "y_axis": {"min": 0.0, "max": 1.0, "label": "Mean cosine similarity"},
                    "reference_lines": [
                        {"value": 0.35, "label": "Quality floor", "color": "red"},
                        {"value": 0.52, "label": "In-distribution mean (observed)", "color": "green"},
                    ],
                },
                {
                    "name": "LLM Quality Score (n≥15 only)",
                    "type": "time_series",
                    "metric": "avg_quality_score",
                    "alert_threshold": None,
                    "description": "LLM-as-judge score (1–5). Visualization only — not alertable at n<15 "
                                   "due to high per-flag variance (20% rate shift per flag at n=5). "
                                   "Enable alerting when sample_size ≥ 15.",
                    "y_axis": {"min": 1.0, "max": 5.0, "label": "Quality score (1–5)"},
                    "reference_lines": [
                        {"value": 3.5, "label": "Acceptable quality floor", "color": "yellow"},
                    ],
                },
                {
                    "name": "Hallucination Rate (n≥15 only)",
                    "type": "time_series",
                    "metric": "hallucination_rate",
                    "alert_threshold": None,
                    "description": "Fraction of responses flagged as hallucinated. Same n≥15 constraint. "
                                   "Leading indicators fire earlier — use this to confirm.",
                    "y_axis": {"min": 0.0, "max": 1.0, "label": "Hallucination rate"},
                    "reference_lines": [
                        {"value": 0.20, "label": "Concern threshold", "color": "yellow"},
                    ],
                },
            ],
        },
        {
            "title": "Operational — Pipeline Health",
            "signal": "latency + saturation",
            "rationale": "Monitor the monitoring system itself. Silent failures here mean drift goes undetected.",
            "panels": [
                {
                    "name": "Batch Processing Latency",
                    "type": "time_series",
                    "metric": "batch_duration_s",
                    "alert_threshold": None,
                    "description": "Time from run_batch() call to check_thresholds() completion. "
                                   "Spike here means the monitoring pipeline is struggling.",
                    "y_axis": {"min": 0, "label": "Seconds"},
                },
                {
                    "name": "Sample Size Per Batch",
                    "type": "time_series",
                    "metric": "sample_size",
                    "alert_threshold": None,
                    "description": "n per batch. Drives statistical power of KS test and LLM judge reliability. "
                                   "Alert if n drops below 15 (LLM judge becomes unreliable) or below 5 (KS noise floor).",
                    "y_axis": {"min": 0, "label": "Sample count"},
                    "reference_lines": [
                        {"value": 15, "label": "LLM judge minimum", "color": "yellow"},
                        {"value": 50, "label": "Target n", "color": "green"},
                    ],
                },
                {
                    "name": "Alert Type Distribution",
                    "type": "bar",
                    "metric": "alert_types",
                    "description": "Count of each alert type fired per rolling window: "
                                   "embedding_drift, centroid_shift, retrieval_degraded. "
                                   "Pattern (embedding before retrieval) confirms leading indicator ordering.",
                },
                {
                    "name": "MLflow Write Success Rate",
                    "type": "stat",
                    "metric": "mlflow_write_success_rate",
                    "alert_threshold": 0.999,
                    "description": "Fraction of batches successfully written to MLflow audit log. "
                                   "Any gap is an audit hole — escalate immediately.",
                    "y_axis": {"min": 0.0, "max": 1.0, "label": "Success rate"},
                },
            ],
        },
    ],
    "alert_routing": {
        "embedding_drift_detection": "oncall",
        "centroid_drift_detection": "oncall",
        "retrieval_quality": "ticket",
        "monitor_latency": "oncall",
        "mlflow_write_success": "ticket",
    },
    "runbook_links": {
        "embedding_drift_detected": "docs/runbooks/embedding-drift.md",
        "retrieval_quality_degraded": "docs/runbooks/retrieval-quality.md",
        "monitor_pipeline_slow": "docs/runbooks/monitor-health.md",
    },
}


def to_markdown(spec: dict) -> str:
    lines = [
        f"# {spec['name']}",
        f"*Generated: {spec['generated_at']}*",
        "",
        spec["description"],
        "",
    ]

    for section in spec["sections"]:
        lines += [
            f"## {section['title']}",
            f"**Signal:** {section['signal']}  ",
            f"**Rationale:** {section['rationale']}",
            "",
        ]
        for panel in section["panels"]:
            thresh = f"Alert threshold: `{panel['alert_threshold']}`" if panel.get("alert_threshold") else "No hard alert"
            lines += [
                f"### {panel['name']}",
                f"- **Type:** {panel['type']}",
                f"- **Metric:** `{panel.get('metric', 'n/a')}`",
                f"- **{thresh}**",
                f"- {panel['description']}",
                "",
            ]

    lines += ["## Alert Routing", ""]
    for metric, dest in spec["alert_routing"].items():
        lines.append(f"- `{metric}` → **{dest}**")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate dashboard specification")
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown")
    parser.add_argument("--export", metavar="FILE", help="Write output to file")
    args = parser.parse_args()

    output = (
        json.dumps(DASHBOARD_SPEC, indent=2)
        if args.format == "json"
        else to_markdown(DASHBOARD_SPEC)
    )

    if args.export:
        Path(args.export).write_text(output)
        print(f"Dashboard spec written to {args.export}")
    else:
        print(output)


if __name__ == "__main__":
    main()
