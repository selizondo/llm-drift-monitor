#!/usr/bin/env python3
"""Design an SLO framework for an LLM monitoring service.

Outputs a complete SLI/SLO/SLA specification with burn rate alert thresholds.
No external dependencies — stdlib only.

Usage:
    python scripts/slo_designer.py
    python scripts/slo_designer.py --service "RAG Pipeline" --target 0.95 --window 7
    python scripts/slo_designer.py --export slo_spec.json
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class SLI:
    name: str
    description: str
    metric_type: str          # leading | lagging | operational
    measurement: str          # what is counted / measured
    good_event_definition: str


@dataclass
class SLO:
    sli: SLI
    target_pct: float         # e.g. 95.0
    window_days: int
    rationale: str

    @property
    def error_budget_pct(self) -> float:
        return 100.0 - self.target_pct

    def error_budget_events(self, total_events: int) -> int:
        return int(total_events * self.error_budget_pct / 100)

    def burn_rate_thresholds(self) -> dict[str, float]:
        """Return fast/slow burn rate multipliers and their alert windows."""
        return {
            "fast_burn_multiplier": 14.4,   # consumes 5% budget in 1h (2% budget/h × 5% = alert in ~1h)
            "fast_burn_window_h": 1,
            "slow_burn_multiplier": 6.0,    # consumes 5% budget in 6h
            "slow_burn_window_h": 6,
            "critical_threshold_multiplier": 2.0,  # page immediately if consuming 2× normal rate
        }


@dataclass
class SLOSpec:
    service_name: str
    generated_at: str
    slos: list[SLO] = field(default_factory=list)

    def add_slo(self, slo: SLO) -> None:
        self.slos.append(slo)


def build_llm_monitoring_slos(service: str, target: float, window: int, batch_cadence_h: float) -> SLOSpec:
    """Build the canonical SLO set for an LLM drift monitoring service."""
    spec = SLOSpec(
        service_name=service,
        generated_at=datetime.now().strftime("%Y-%m-%d"),
    )

    _batches_per_window = int((window * 24) / batch_cadence_h)

    slos = [
        SLO(
            sli=SLI(
                name="embedding_drift_detection",
                description="KS test on top-variance embedding dimensions fires within acceptable threshold",
                metric_type="leading",
                measurement="KS statistic < 0.3 at p < 0.05 per batch",
                good_event_definition="Batch KS stat below threshold OR p-value above 0.05 (no significant drift)",
            ),
            target_pct=target * 100,
            window_days=window,
            rationale="Leading indicator: fires 2-3 batches before retrieval quality degrades. High target justified by low false-positive rate after empirical threshold calibration.",
        ),
        SLO(
            sli=SLI(
                name="centroid_drift_detection",
                description="Semantic centroid cosine distance within acceptable range",
                metric_type="leading",
                measurement="Centroid cosine distance < 0.30 from baseline",
                good_event_definition="Per-batch centroid distance below 0.30 threshold",
            ),
            target_pct=target * 100,
            window_days=window,
            rationale="Complements KS test: detects directional distribution shift vs. spread change.",
        ),
        SLO(
            sli=SLI(
                name="retrieval_quality",
                description="Mean retrieval similarity score stays above minimum quality floor",
                metric_type="leading",
                measurement="Mean cosine similarity >= 0.35 per batch",
                good_event_definition="Per-batch mean retrieval similarity at or above 0.35",
            ),
            target_pct=(target - 0.05) * 100,
            window_days=window,
            rationale="Slightly lower target than embedding SLOs — retrieval quality has higher natural variance across query types.",
        ),
        SLO(
            sli=SLI(
                name="monitor_latency",
                description="Alert emitted within 2 batch cycles of threshold breach",
                metric_type="operational",
                measurement="Time from batch arrival to alert emission <= 2 × batch_cadence_h",
                good_event_definition="Alert (if warranted) emitted within 2 × batch cadence",
            ),
            target_pct=99.0,
            window_days=window,
            rationale="Monitoring pipeline must be reliable. If it fails silently, drift goes undetected.",
        ),
        SLO(
            sli=SLI(
                name="mlflow_write_success",
                description="All completed batches written to MLflow audit log",
                metric_type="operational",
                measurement="MLflow write success rate per batch",
                good_event_definition="Batch record written to MLflow without error",
            ),
            target_pct=99.9,
            window_days=window,
            rationale="Audit completeness is a hard requirement — gaps in the log undermine incident investigations.",
        ),
    ]

    for slo in slos:
        spec.add_slo(slo)

    return spec


def print_spec(spec: SLOSpec) -> None:
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"SLO SPECIFICATION — {spec.service_name}")
    print(f"Generated: {spec.generated_at}")
    print(sep)

    for i, slo in enumerate(spec.slos, 1):
        eb = 100.0 - slo.target_pct
        burns = slo.burn_rate_thresholds()
        print(f"\n[SLO {i}] {slo.sli.name.upper().replace('_', ' ')}")
        print(f"  SLI:        {slo.sli.description}")
        print(f"  Measure:    {slo.sli.measurement}")
        print(f"  Good event: {slo.sli.good_event_definition}")
        print(f"  Type:       {slo.sli.metric_type}")
        print(f"  Target:     {slo.target_pct:.1f}% over {slo.window_days} days")
        print(f"  Error budget: {eb:.1f}%  ({eb * slo.window_days * 24 / 100:.1f} hours in window)")
        print(f"  Rationale:  {slo.rationale}")
        print("  Burn rate alerts:")
        print(f"    Fast burn: >{burns['fast_burn_multiplier']}x rate → page now "
              f"(window: {burns['fast_burn_window_h']}h)")
        print(f"    Slow burn: >{burns['slow_burn_multiplier']}x rate → ticket "
              f"(window: {burns['slow_burn_window_h']}h)")

    print(f"\n{sep}")
    print("GOLDEN SIGNALS MAPPING")
    print(sep)
    signals = {
        "Latency": "Time from run_batch() call to check_thresholds() completion",
        "Traffic": "Batches per hour; sample size (n) per batch",
        "Errors":  "MLflow write failures · W&B API errors · embedding model unavailable",
        "Saturation": "Baseline array RAM usage · embedding model queue depth",
    }
    for signal, description in signals.items():
        print(f"  {signal:<12} {description}")

    print(f"\n{sep}")
    print("ALERT ROUTING")
    print(sep)
    print("  embedding_drift_detection  → oncall (leading indicator, high urgency)")
    print("  centroid_drift_detection   → oncall (leading indicator, high urgency)")
    print("  retrieval_quality          → ticket (lagging, investigate within 24h)")
    print("  monitor_latency            → oncall (monitoring is broken)")
    print("  mlflow_write_success       → ticket (audit gap, non-urgent)")


def export_json(spec: SLOSpec, path: str) -> None:
    output = {
        "service": spec.service_name,
        "generated_at": spec.generated_at,
        "slos": [
            {
                "name": slo.sli.name,
                "type": slo.sli.metric_type,
                "target_pct": slo.target_pct,
                "window_days": slo.window_days,
                "error_budget_pct": slo.error_budget_pct,
                "measurement": slo.sli.measurement,
                "good_event": slo.sli.good_event_definition,
                "rationale": slo.rationale,
                "burn_rate": slo.burn_rate_thresholds(),
            }
            for slo in spec.slos
        ],
    }
    Path = __import__("pathlib").Path
    Path(path).write_text(json.dumps(output, indent=2))
    print(f"\nExported to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Design SLO framework for LLM monitoring")
    parser.add_argument("--service", default="LLM Drift Monitor", help="Service name")
    parser.add_argument("--target", type=float, default=0.95, help="SLO target (0–1, default 0.95)")
    parser.add_argument("--window", type=int, default=7, help="Rolling window in days")
    parser.add_argument("--cadence", type=float, default=1.0, help="Batch cadence in hours")
    parser.add_argument("--export", metavar="FILE", help="Export spec to JSON file")
    args = parser.parse_args()

    if not 0 < args.target < 1:
        print("--target must be between 0 and 1 (e.g. 0.95 for 95%)")
        sys.exit(1)

    spec = build_llm_monitoring_slos(args.service, args.target, args.window, args.cadence)
    print_spec(spec)

    if args.export:
        export_json(spec, args.export)


if __name__ == "__main__":
    main()
