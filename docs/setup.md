# Setup and Usage

## Key Concepts

**Embedding drift (KS test):** Kolmogorov-Smirnov test on the top-20 variance dimensions of query embeddings. Detects semantic distribution shift. Accounts for sample size automatically: p-value is valid at any n. KS fires on batch 6 when topics shift from ML Q&A to manufacturing/finance/healthcare; fires **before** quality scores drop (batch 6 vs batch 8), giving ops time to investigate.

**Centroid drift:** Cosine distance between the query embedding centroid of the current batch and the baseline (batches 1-5). Directional semantic shift: even if individual queries embed fine, their centroid can drift. Measured 0.002 in-distribution, 0.0891 out-of-distribution.

**Leading vs lagging signals:** Embedding drift (KS + centroid) fires first. Quality scores are a lagging signal: LLMs handle short OOD questions okay until retrieval context collapses. Operational insight: embedding drift gives you response time; accuracy gives you a post-mortem.

**MLflow for audit, W&B for live monitoring:** Separate concerns, separate tools. MLflow answers "what happened in batch 6?" (diffs, diffs, rollback decisions). W&B answers "what is happening now?" (live dashboard, alert thresholds). No cross-importing between modules — clean ownership boundary.

**Statistical grounding for thresholds:** KS and centroid metrics are threshold-based. Why KS over PSI? PSI requires n≥1000 per bucket; KS is valid at any n. Quality alerts disable below n=15: at n=5, one hallucination flag shifts the rate by 20%, indistinguishable from genuine degradation.

---

## Prerequisites

- Python 3.10+
- `ANTHROPIC_API_KEY` in environment (or use `--no-quality` to skip LLM scoring)
- `WANDB_API_KEY` in environment (or use `--no-wandb` to skip dashboard)

## Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Generate 10 batches (50 queries each; batches 6-8 inject OOD queries)
python data/simulate_stream.py

# 3. Drift detection only (no API keys required)
python run_monitor.py --no-quality --no-wandb

# 4. Full run with quality scoring and W&B dashboard
export ANTHROPIC_API_KEY=...
export WANDB_API_KEY=...
python run_monitor.py

# 5. Inspect audit trail
mlflow ui   # http://127.0.0.1:5000
```

## What You Should See

Three phases visible in the terminal output and W&B dashboard:

```
Batch 01 [OK   ]
  drift  : 0% dims | PSI=0.012 | centroid=0.0021
  quality: score=2.85/3 | halluc=0%

...

Batch 06 [ALERT]
  drift  : 75% dims | PSI=0.341 | centroid=0.0891
  quality: score=1.60/3 | halluc=40%
  ! embedding_drift: pct_dims_drifted=75% (threshold 30%)
  ! length_drift: psi_query_length=0.341 (threshold 0.25)
  ! centroid_drift: centroid_drift=0.0891 (threshold 0.05)
  ! quality_degraded: avg_quality_score=1.60 (threshold 2.0)

...

Batch 09 [OK   ]
  drift  : 5% dims | PSI=0.018 | centroid=0.0019
  quality: score=2.80/3 | halluc=0%
```

## Drift Simulation

| Batches | Content | Domain |
|---------|---------|--------|
| 1 to 5 | ML Q&A (from lora-finetune val set) | in-distribution |
| 6 to 8 | 70% out-of-domain + 30% ML Q&A | drift window |
| 9 to 10 | ML Q&A only | recovered |

Out-of-domain queries span manufacturing (PLC, SCADA), finance (risk, trading), healthcare (clinical trials, EHR), and supply chain. They share vocabulary with ML Q&A but are semantically distant: exactly the kind of drift hard to catch without monitoring.

## Detection Signals

| Signal | Method | Fires on |
|--------|--------|----------|
| Embedding drift | KS test (top-20 variance dims) | Semantic distribution shift |
| Length drift | PSI on word count bins | Query pattern change |
| Centroid drift | Cosine distance from baseline | Directional semantic shift |
| Quality degradation | Self-judge (1 to 3 scale, n>=15) | Retrieval failure producing poor answers |
| Hallucination rate | Judge flag (n>=15) | Unsupported claims in answers |

Quality degradation is structural, not simulated. OOD queries fail cosine retrieval (similarity below 0.3) against the ML corpus. The model answers without context. Judge scores drop naturally.

## Code Layout

```
llm-drift-monitor/
├── data/
│   ├── simulate_stream.py   # Build 10 batches; inject OOD queries in batches 6-8
│   └── batches/             # Generated: batch_01.jsonl ... batch_10.jsonl
├── monitor/
│   ├── embeddings.py        # embed_queries(), compute_baseline(), centroid_drift()
│   ├── drift.py             # KS test + PSI; compute_drift_report()
│   ├── quality.py           # Retrieve -> generate -> self-judge; score_batch_sample()
│   ├── trends.py            # check_thresholds(), summarize_batch()
│   ├── logger.py            # MLflow: log_baseline(), log_batch()
│   └── dashboard.py         # W&B: init_run(), log_batch(), finish()
└── run_monitor.py           # Entry point: build baseline -> loop over batches -> log
```
