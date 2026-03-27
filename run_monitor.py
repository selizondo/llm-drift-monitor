"""
run_monitor.py — Main entry point: run the monitoring pipeline over all batches.

Loads batches from data/batches/, computes drift + quality signals per batch,
logs everything to MLflow (audit trail) and W&B (live dashboard).

Usage:
    python run_monitor.py                              # full run, all 10 batches
    python run_monitor.py --batches 10 --no-quality    # skip LLM quality scoring
    python run_monitor.py --no-wandb                   # MLflow only
    python run_monitor.py --quality-sample 3           # sample 3 queries per batch
    python run_monitor.py --ollama-quality llama3.2    # Ollama backend (no API key)

Prerequisites:
    pip install -r requirements.txt
    python data/simulate_stream.py       # generates data/batches/
    export ANTHROPIC_API_KEY=...         # required unless --no-quality or --ollama-quality
    export WANDB_API_KEY=...             # required unless --no-wandb
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

import numpy as np

from monitor import embeddings as emb_module
from monitor import logger as mlf_logger
from monitor import dashboard as wb_dashboard
from monitor.drift import compute_drift_report
from monitor.quality import score_batch_sample, retrieval_quality_score
from monitor.trends import check_thresholds, summarize_batch

BATCHES_DIR = Path("data/batches")

# Corpus version tag — bump when embedding model or corpus changes so runs are
# comparable. Surfaced in the JSON summary so consumers can detect version drift.
CORPUS_VERSION = os.getenv("CORPUS_VERSION", "v1")


def load_batch(batch_num: int) -> list[dict]:
    path = BATCHES_DIR / f"batch_{batch_num:02d}.jsonl"
    if not path.exists():
        sys.exit(f"Batch file not found: {path}\nRun: python data/simulate_stream.py")
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM production drift monitor")
    parser.add_argument("--batches", type=int, default=10, help="Number of batches to process")
    parser.add_argument("--quality-sample", type=int, default=5, help="Queries sampled per batch for quality scoring")
    parser.add_argument("--no-wandb", action="store_true", help="Skip W&B logging")
    parser.add_argument("--no-quality", action="store_true", help="Skip quality scoring (no API calls)")
    parser.add_argument(
        "--ollama-quality",
        metavar="MODEL",
        default=None,
        help=(
            "Use a local Ollama model for quality scoring instead of Anthropic (e.g. llama3.2). "
            "No ANTHROPIC_API_KEY required. Requires Ollama running at localhost:11434."
        ),
    )
    args = parser.parse_args()

    # -- Setup --
    mlf_logger.init_experiment()

    if not args.no_wandb:
        wb_dashboard.init_run()

    # Quality backend: Anthropic (default) or Ollama (--ollama-quality MODEL).
    # WHY separate from --no-quality: --ollama-quality enables quality scoring
    # without an API key; --no-quality disables it entirely regardless of backend.
    anthropic_client = None
    ollama_quality_model: str | None = None

    if not args.no_quality:
        if args.ollama_quality:
            ollama_quality_model = args.ollama_quality
        elif os.environ.get("ANTHROPIC_API_KEY"):
            import anthropic
            anthropic_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        else:
            sys.exit(
                "ANTHROPIC_API_KEY not set. "
                "Use --no-quality to skip quality scoring, "
                "or --ollama-quality MODEL to use a local Ollama model."
            )

    # -- Baseline (batch 1) --
    print("Building baseline from batch 1...")
    baseline_records = load_batch(1)
    baseline_queries = [r["query"] for r in baseline_records]
    baseline_embeddings = emb_module.embed_queries(baseline_queries)
    baseline_stats = emb_module.compute_baseline(baseline_embeddings)
    baseline_lengths = [len(q.split()) for q in baseline_queries]
    baseline_centroid = np.array(baseline_stats["centroid"])
    mlf_logger.log_baseline(baseline_stats)
    print(f"  {len(baseline_queries)} queries | {baseline_embeddings.shape[1]}-dim embeddings\n")
    print("-" * 62)

    # -- Monitoring loop --
    all_alerts: list[dict] = []
    for batch_num in range(1, args.batches + 1):
        records = load_batch(batch_num)
        queries = [r["query"] for r in records]

        current_embeddings = emb_module.embed_queries(queries)
        current_lengths = [len(q.split()) for q in queries]

        centroid_d = emb_module.centroid_drift(baseline_centroid, current_embeddings)
        drift_report = compute_drift_report(
            baseline_embeddings, current_embeddings, baseline_lengths, current_lengths
        )
        drift_report["centroid_drift"] = round(centroid_d, 4)

        # Retrieval quality — always computed, no API calls needed
        retrieval = retrieval_quality_score(queries, emb_module)

        # LLM judge quality — optional, requires Anthropic API key or local Ollama
        llm_quality = {"avg_quality_score": 0.0, "hallucination_rate": 0.0, "n_sampled": 0}
        if anthropic_client or ollama_quality_model:
            llm_quality = score_batch_sample(
                queries,
                emb_module,
                sample_size=args.quality_sample,
                client=anthropic_client,
                ollama_model=ollama_quality_model,
            )

        metrics = {
            "pct_dims_drifted": drift_report["pct_dims_drifted"],
            "ks_length_p_value": drift_report["ks_length_p_value"],
            "ks_length_drifted": drift_report["ks_length_drifted"],
            "psi_query_length": drift_report["psi_query_length"],
            "centroid_drift": centroid_d,
            "avg_retrieval_sim": retrieval["avg_retrieval_sim"],
            "retrieval_miss_rate": retrieval["retrieval_miss_rate"],
            "avg_quality_score": llm_quality["avg_quality_score"],
            "hallucination_rate": llm_quality["hallucination_rate"],
            "n_sampled": llm_quality["n_sampled"],
        }
        alerts = check_thresholds(metrics, llm_judge_enabled=bool(anthropic_client or ollama_quality_model))

        if alerts:
            all_alerts.append({"batch": batch_num, "alerts": alerts})

        mlf_logger.log_batch(batch_num, metrics, alerts, drift_report)
        if not args.no_wandb:
            wb_dashboard.log_batch(batch_num, metrics, alerts)

        print(summarize_batch(batch_num, metrics, alerts))
        print()

    # -- Finish --
    if not args.no_wandb:
        wb_dashboard.finish()

    print("-" * 62)
    print(f"MLflow  : run `mlflow ui` then open http://127.0.0.1:5000")
    if not args.no_wandb:
        entity = os.environ.get("WANDB_ENTITY", "<your-entity>")
        print(f"W&B     : https://wandb.ai/{entity}/llm-drift-monitor")

    # -- JSON summary (structured output for downstream consumers / CI) --
    # Writes triggered_alerts to a JSON file so CI scripts and eval harnesses can
    # consume the run summary without parsing stdout or connecting to MLflow/W&B.
    summary_path = Path("artifacts/run_summary.json")
    summary_path.parent.mkdir(exist_ok=True)
    summary_path.write_text(json.dumps({
        "corpus_version": CORPUS_VERSION,
        "batches_processed": args.batches,
        "triggered_alerts": all_alerts,
        "judge_backend": "ollama" if args.ollama_quality else ("anthropic" if not args.no_quality else "none"),
    }, indent=2))
    print(f"Summary : {summary_path}")


if __name__ == "__main__":
    main()
