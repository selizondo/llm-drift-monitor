# Design and Tradeoffs

Decisions made during build, with the reasoning and scale or complexity boundary where each breaks down.

---

## ADR-01: MLflow vs W&B: Hard Ownership Boundary

**Decision:** Both tools, with strict separation. MLflow = audit trail. W&B = live time-series dashboard. Neither module imports from the other.

| Tool | Owns | Answers |
|------|------|---------|
| MLflow | Per-batch run records, drift report artifacts, baseline snapshot, alert tags | "What happened in batch 6?" |
| W&B | Time-series, cross-batch trend lines, alert threshold lines | "What is happening now?" |

**Why not one tool?** MLflow's artifact model is excellent for audit but its visualization layer is not built for live trend monitoring. W&B's run history is optimized for visualization, not structured querying ("show all batches where alert_types = 'embedding_drift'"). Each tool does what it's good at. Cost: two thin modules (~30 lines each).

**Why not Prometheus + Grafana?** Right architecture for production at scale. Setup overhead (cluster, exporters, PromQL) is disproportionate to the demo. Listed in product.md as the production upgrade path.

**Why not Evidently AI?** Purpose-built for data drift with rich HTML reports. Does not provide time-series dashboard capability and adds a third tool. Right choice for richer drift reports in a production system.

**Tradeoff:** Two accounts, two API keys, two places to look. A batch with no W&B alerts may still have useful detail in MLflow. Requires knowing which tool to check for which question. This ADR is that documentation.

**Scale boundary:** At production load (100+ batches/day), MLflow becomes a query bottleneck unless using a remote tracking server (Postgres/MySQL backend). W&B handles volume natively.

---

## KS Test Over PSI for Alerting

**Decision:** KS test for drift alerts. PSI logged for trend visualization but not used for hard alerts.

**Why:** PSI requires large samples to be reliable (industry calibration assumes n of 1000 or more per bucket). At n=50 per batch, PSI produces high variance. KS test accounts for sample size automatically: the p-value is valid at any n.

**Tradeoff:** KS is a rank test (ignores the shape of the distribution within ranks). PSI is more interpretable in absolute terms (PSI below 0.1 = stable; 0.1 to 0.25 = watch; above 0.25 = alert). Both are logged; only KS generates alerts.

**Scale boundary:** At n of 500 or more per batch, PSI becomes reliable and can replace or supplement KS. The standard industry thresholds are calibrated for this sample size.

---

## Top-20 Embedding Dimensions for KS Test

**Decision:** Test KS on the 20 highest-variance dimensions of the baseline embeddings rather than all 384.

**Why:** All-MiniLM-L6-v2 produces 384-dim embeddings, but many dimensions have near-zero variance and carry no distributional signal. The top-20 by variance capture the semantic axes most likely to shift when query distribution changes. Testing all 384 dims increases multiple-comparison risk and noise without improving detection.

**Tradeoff:** A shift in low-variance dimensions is not caught. In practice, distributional shifts that matter (OOD queries, domain drift) show up in high-variance dims first.

**Scale boundary:** With a larger baseline corpus (n of 500 or more), PCA or tSNE could identify a more principled set of informative dims. For n=50 baseline, variance-sort is the practical choice.

---

## Leading vs Lagging Indicators

**Decision:** Embedding drift signals (KS, centroid cosine) are primary and leading. LLM quality scores are secondary and lagging. Both are monitored; alerting prioritizes the leading indicators.

**Why:** The LLM handles short OOD questions adequately even without retrieved context: quality scores degrade slowly. Embedding centroid drift and retrieval similarity move earlier because they directly measure semantic distance from the training distribution. Alerting on leading indicators gives more response time.

**Tradeoff:** Leading indicators have false-positive risk. A batch of unusual-but-valid ML questions may trigger embedding drift without quality degradation. The design accepts this: false positives are preferable to late detection.

---

## LLM Judge Alerts Disabled at n=5

**Decision:** `check_thresholds()` does not fire alerts on `avg_quality_score` or `hallucination_rate` when `n_sampled < 15`. Both are logged to MLflow and W&B for trend visualization regardless of sample size.

**Why:** At n=5 queries per batch, each hallucination flag shifts the rate by 20%. A single outlier response produces `hallucination_rate = 0.20`, indistinguishable from genuine degradation. Hard alerts at n=5 would fire constantly on noise. The default `sample_size=5` is a cost and latency choice for the demo, not a statistical choice.

**Tradeoff:** Genuine quality degradation in LLM output goes undetected unless retrieval similarity also drops. The assumption: retrieval similarity is a reliable proxy. If the model lacks relevant context, quality will degrade. This is a documented assumption, not a guarantee.

**Scale boundary (production):** At n of 15 or more per batch, each hallucination flag shifts the rate by 6.7%: reliable enough for hard alerting. Pass `--quality-sample 15` (or higher) to `run_monitor.py` to enable LLM judge alerts.

---

## Retrieval Corpus: Inter-Repo Dependency

**Decision:** `quality.py` uses `lora-finetune/data/train.jsonl` as the retrieval corpus via a relative path three levels up in the monorepo.

**Why:** The lora-finetune training data is StackOverflow ML Q&A, the same distribution as the simulated batch queries. In-distribution queries score high similarity; OOD queries score low. This is exactly the signal needed.

**Tradeoff:** Path resolves only in the monorepo layout. Cloning `llm-drift-monitor` independently breaks `CORPUS_PATH.exists()`. Applied fix: `quality.py` logs a warning at import time if `CORPUS_PATH` does not exist. Non-fatal: the monitor continues with retrieval scores of 0.0.

---

## SLO Framework

Applying the SLI/SLO/SLA pattern explicitly to define what "working correctly" means and when to page.

**Service Level Indicators:**

| Signal | Indicator | Type |
|--------|-----------|------|
| Embedding drift | KS statistic on top-20 dims per batch | Leading |
| Semantic drift | Cosine distance of batch centroid from baseline | Leading |
| Retrieval quality | Mean retrieval similarity score per batch | Leading |
| Output quality | Average LLM judge score (n of 15 or more per batch) | Lagging |
| Hallucination rate | Fraction of responses flagged per batch (n of 15 or more) | Lagging |

**Service Level Objectives:**

| SLI | Target | Window |
|-----|--------|--------|
| KS stat (embedding) | Below 0.3 at p below 0.05 for 95% of batches | Rolling 7-day |
| Centroid cosine dist | Below 0.3 for 95% of batches | Rolling 7-day |
| Retrieval similarity | 0.35 or higher mean for 90% of batches | Rolling 7-day |
| Monitor latency | Alert emitted within 2 batches of threshold breach | Per-breach |

**Burn rate alert design:** At hourly batch cadence, the 7-day error budget = 168 batches x 5% = 8.4 allowed failures. Fast burn alert (2x rate): if 4 or more failures occur in 1 hour, page immediately. Slow burn alert: if 6 or more failures occur in 24 hours, escalate to on-call review.

---

## What Was Cut

| Cut | Reason | Upgrade trigger |
|-----|--------|-----------------|
| Prometheus + Grafana | Setup overhead disproportionate to demo | Production deployment at scale |
| Evidently AI drift reports | Adds third tool, less time-series capability | Rich drift HTML reports for stakeholders |
| Async batch processing | Single-threaded simpler for demo | More than 10 concurrent batch streams |
| Model re-indexing on drift | Out of scope: this project monitors, not remediates | Automated remediation pipeline |
| Hard LLM judge alerts at n=5 | Too noisy at that sample size | Increase sample_size to 15 or more |

---

## Architectural Standard

The design decision here: separating audit from alerting into two tools with a documented ownership boundary means any team can stand up production ML observability in a day rather than reinventing the split from scratch. The MLflow + W&B split is not a personal preference. It is a reusable pattern: one tool owns the immutable audit trail, one tool owns live alerting, and the boundary is documented in this ADR so the next team that asks "should we use both?" has a written answer instead of a debate.

The leading-indicator design (KS drift fires at batch 6; quality scores drop at batch 8) is the key operational insight: PSI and KS are signals you can act on before users are affected. Accuracy is a lagging indicator you discover after. Any monitoring system that only tracks accuracy is reactive by design. The boundary between proactive and reactive monitoring is the threshold tuning methodology: documented here, not left as tribal knowledge.
