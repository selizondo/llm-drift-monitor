#!/usr/bin/env python3
"""Analyze historical drift batches and recommend optimized alert thresholds.

Reads from the MLflow SQLite database (mlflow.db) and computes:
- False positive rate at current thresholds
- Recommended thresholds that minimize noise while preserving signal
- Burn rate configuration for multi-window alerting

No external dependencies — uses stdlib sqlite3.

Usage:
    python scripts/alert_optimizer.py                        # reads mlflow.db
    python scripts/alert_optimizer.py --db path/to/mlflow.db
    python scripts/alert_optimizer.py --export thresholds.json
    python scripts/alert_optimizer.py --demo               # synthetic data demo
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import statistics
import sys
from pathlib import Path

DEFAULT_DB = Path(__file__).parent.parent / "mlflow.db"

# Current thresholds from docs/tradeoffs.md — empirically calibrated values.
CURRENT_THRESHOLDS = {
    "ks_stat": 0.30,
    "centroid_dist": 0.30,
    "retrieval_sim_min": 0.35,
}


def load_from_mlflow(db_path: Path) -> list[dict]:
    """Extract batch metrics from MLflow SQLite database."""
    if not db_path.exists():
        return []

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # MLflow stores metrics in the 'metrics' table with run_uuid, key, value, step.
    try:
        cursor.execute("""
            SELECT r.run_uuid, r.start_time, m.key, m.value
            FROM runs r
            JOIN metrics m ON r.run_uuid = m.run_uuid
            WHERE m.key IN ('ks_stat_max', 'centroid_dist', 'mean_retrieval_sim',
                            'alert_triggered', 'batch_num', 'sample_size')
            ORDER BY r.start_time, m.key
        """)
        rows = cursor.fetchall()
    except sqlite3.OperationalError:
        conn.close()
        return []

    conn.close()

    # Pivot: group metric values by run_uuid.
    runs: dict[str, dict] = {}
    for row in rows:
        uid = row["run_uuid"]
        if uid not in runs:
            runs[uid] = {"run_uuid": uid, "start_time": row["start_time"]}
        runs[uid][row["key"]] = row["value"]

    return list(runs.values())


def make_synthetic_batches(n: int = 30) -> list[dict]:
    """Generate synthetic batch data: 25 in-distribution, 5 OOD."""
    import random
    random.seed(42)
    batches = []
    for i in range(n):
        is_ood = i >= 25
        batches.append({
            "batch_num": i + 1,
            "ks_stat_max": random.gauss(0.35 if is_ood else 0.12, 0.05),
            "centroid_dist": random.gauss(0.48 if is_ood else 0.08, 0.04),
            "mean_retrieval_sim": random.gauss(0.28 if is_ood else 0.52, 0.05),
            "alert_triggered": is_ood,
            "sample_size": 50,
        })
    return batches


def analyze_thresholds(batches: list[dict], thresholds: dict) -> dict:
    """Compute false positive / detection metrics at given thresholds."""
    ks_thresh = thresholds["ks_stat"]
    cent_thresh = thresholds["centroid_dist"]
    ret_thresh = thresholds["retrieval_sim_min"]

    true_alerts = sum(1 for b in batches if b.get("alert_triggered"))
    total = len(batches)
    non_alerts = total - true_alerts

    false_positives = 0
    true_positives = 0
    false_negatives = 0

    for b in batches:
        ks = b.get("ks_stat_max", 0)
        cent = b.get("centroid_dist", 0)
        ret = b.get("mean_retrieval_sim", 1)

        fired = ks > ks_thresh or cent > cent_thresh or ret < ret_thresh
        actual = bool(b.get("alert_triggered"))

        if fired and not actual:
            false_positives += 1
        elif fired and actual:
            true_positives += 1
        elif not fired and actual:
            false_negatives += 1

    return {
        "total_batches": total,
        "true_alert_batches": true_alerts,
        "false_positives": false_positives,
        "true_positives": true_positives,
        "false_negatives": false_negatives,
        "false_positive_rate": false_positives / max(non_alerts, 1),
        "recall": true_positives / max(true_alerts, 1),
        "precision": true_positives / max(true_positives + false_positives, 1),
    }


def suggest_thresholds(batches: list[dict]) -> dict[str, float]:
    """Suggest thresholds at mean + 2σ for in-distribution batches."""
    normal = [b for b in batches if not b.get("alert_triggered")]
    if len(normal) < 5:
        return CURRENT_THRESHOLDS

    def safe_stat(key: str, higher_is_alert: bool) -> float:
        vals = [b[key] for b in normal if key in b]
        if not vals:
            return CURRENT_THRESHOLDS.get(key, 0.3)
        mu, sigma = statistics.mean(vals), statistics.stdev(vals) if len(vals) > 1 else 0.05
        return round(mu + 2 * sigma if higher_is_alert else mu - 2 * sigma, 3)

    return {
        "ks_stat": safe_stat("ks_stat_max", higher_is_alert=True),
        "centroid_dist": safe_stat("centroid_dist", higher_is_alert=True),
        "retrieval_sim_min": safe_stat("mean_retrieval_sim", higher_is_alert=False),
    }


def burn_rate_config(window_days: int, target_pct: float, cadence_h: float) -> dict:
    """Compute burn rate alert parameters."""
    total_batches = int(window_days * 24 / cadence_h)
    error_budget = total_batches * (1 - target_pct)
    return {
        "window_days": window_days,
        "target_pct": target_pct,
        "total_batches_in_window": total_batches,
        "error_budget_batches": round(error_budget, 1),
        "fast_burn": {
            "description": "Consuming error budget 14.4x faster than normal → page immediately",
            "threshold_batches_per_hour": round(error_budget * 14.4 / (window_days * 24), 2),
            "alert_window_h": 1,
        },
        "slow_burn": {
            "description": "Consuming error budget 6x faster than normal → create ticket",
            "threshold_batches_per_hour": round(error_budget * 6 / (window_days * 24), 2),
            "alert_window_h": 6,
        },
    }


def print_report(batches: list[dict], source: str) -> dict:
    sep = "=" * 68
    print(f"\n{sep}")
    print(f"ALERT THRESHOLD OPTIMIZER — {source}")
    print(f"Batches analyzed: {len(batches)}")
    print(sep)

    current_metrics = analyze_thresholds(batches, CURRENT_THRESHOLDS)
    suggested = suggest_thresholds(batches)
    suggested_metrics = analyze_thresholds(batches, suggested)

    print("\nCURRENT THRESHOLDS")
    for k, v in CURRENT_THRESHOLDS.items():
        print(f"  {k:<26} {v:.3f}")

    print("\nPERFORMANCE AT CURRENT THRESHOLDS")
    for label, val in [
        ("False positive rate", f"{current_metrics['false_positive_rate']:.1%}"),
        ("Recall (true alerts caught)", f"{current_metrics['recall']:.1%}"),
        ("Precision", f"{current_metrics['precision']:.1%}"),
        ("False positives", str(current_metrics["false_positives"])),
        ("False negatives (missed)", str(current_metrics["false_negatives"])),
    ]:
        print(f"  {label:<32} {val}")

    print("\nSUGGESTED THRESHOLDS (mean + 2σ on in-distribution batches)")
    for k in CURRENT_THRESHOLDS:
        curr = CURRENT_THRESHOLDS[k]
        sugg = suggested[k]
        delta = sugg - curr
        direction = "↑ looser" if delta > 0 else "↓ tighter"
        print(f"  {k:<26} {sugg:.3f}  ({direction:10s} Δ{abs(delta):.3f} from {curr:.3f})")

    print("\nPERFORMANCE AT SUGGESTED THRESHOLDS")
    for label, val in [
        ("False positive rate", f"{suggested_metrics['false_positive_rate']:.1%}"),
        ("Recall (true alerts caught)", f"{suggested_metrics['recall']:.1%}"),
        ("Precision", f"{suggested_metrics['precision']:.1%}"),
    ]:
        print(f"  {label:<32} {val}")

    print("\nBURN RATE CONFIGURATION")
    br = burn_rate_config(window_days=7, target_pct=0.95, cadence_h=1.0)
    print(f"  Error budget:  {br['error_budget_batches']} batches in {br['window_days']}-day window")
    print(f"  Fast burn:     >{br['fast_burn']['threshold_batches_per_hour']}/h in {br['fast_burn']['alert_window_h']}h → PAGE")
    print(f"  Slow burn:     >{br['slow_burn']['threshold_batches_per_hour']}/h in {br['slow_burn']['alert_window_h']}h → TICKET")

    print(sep)
    return {"current": CURRENT_THRESHOLDS, "suggested": suggested, "burn_rate": br}


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize drift detection alert thresholds")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="MLflow SQLite database path")
    parser.add_argument("--demo", action="store_true", help="Use synthetic data (no MLflow needed)")
    parser.add_argument("--export", metavar="FILE", help="Export recommendations to JSON")
    args = parser.parse_args()

    if args.demo:
        batches = make_synthetic_batches()
        source = "synthetic demo data (25 normal + 5 OOD batches)"
    else:
        batches = load_from_mlflow(args.db)
        if not batches:
            print(f"No batch data found in {args.db}. Run the monitor first, or use --demo.")
            sys.exit(1)
        source = f"MLflow: {args.db}"

    result = print_report(batches, source)

    if args.export:
        Path(args.export).write_text(json.dumps(result, indent=2))
        print(f"\nExported to {args.export}")


if __name__ == "__main__":
    main()
