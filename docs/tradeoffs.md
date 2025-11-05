# Architectural Tradeoffs

Decisions made during build, with the reasoning and scale/complexity boundaries.

---

## MLflow vs W&B — Hard Ownership Boundary

**Decision:** Both tools, with strict separation. MLflow = audit trail; W&B = live time-series dashboard. Neither module imports from the other.

| Tool | Owns | Answers |
|---|---|---|
| **MLflow** | Per-batch run records, drift report artifacts, baseline snapshot, alert tags | "What happened in batch 6?" |
| **W&B** | Time-series, cross-batch trend lines, alert threshold lines | "What is happening now?" |

**Why not one tool?** MLflow's artifact model is excellent for audit but its visualization layer isn't built for live trend monitoring. W&B's run history is optimized for visualization, not structured querying ("show all batches where alert_types = 'embedding_drift'"). Each tool does what it's good at; cost is two thin modules (~30 lines each).

**Why not Prometheus + Grafana?** Right architecture for production at scale. Setup overhead (cluster, exporters, PromQL) is disproportionate to the demo. Listed in README as the production upgrade path.

**Why not Evidently AI?** Purpose-built for data drift with rich HTML reports. Doesn't provide time-series dashboard capability and adds a third tool. Right choice for richer drift reports in a production system.

**Tradeoff:** Two accounts, two API keys, two places to look. A batch with no W&B alerts may still have useful detail in MLflow — requires knowing to check both.

**Scale boundary:** At production load (100+ batches/day), MLflow becomes a query bottleneck unless using a remote tracking server (Postgres/MySQL backend). W&B handles volume natively; MLflow needs ops to scale.

---

## KS Test Over PSI for Alerting

**Decision:** KS test for drift alerts; PSI logged for trend visualization but not used for hard alerts.

**Why:** PSI requires large samples to be reliable (industry calibration assumes n≥1000 per bucket). At n=50 per batch, PSI produces high variance. KS test accounts for sample size automatically — the p-value is valid at any n.

**Tradeoff:** KS is a rank test (ignores the shape of the distribution within ranks). PSI is more interpretable in absolute terms (PSI < 0.1 = stable; 0.1–0.25 = watch; > 0.25 = alert). Both are logged; only KS generates alerts.

**Scale boundary:** At n≥500 per batch, PSI becomes reliable and can replace or supplement KS. The standard industry thresholds (0.1/0.25) are calibrated for this sample size.

---

## Top-20 Embedding Dimensions for KS Test

**Decision:** Test KS on the 20 highest-variance dimensions of the baseline embeddings rather than all 384.

**Why:** All-MiniLM-L6-v2 produces 384-dim embeddings, but many dimensions have near-zero variance and carry no distributional signal. The top-20 by variance capture the semantic axes most likely to shift when query distribution changes. Testing all 384 dims increases multiple-comparison risk and noise without improving detection.

**Tradeoff:** A shift in low-variance dimensions is not caught. In practice, distributional shifts that matter (OOD queries, domain drift) show up in high-variance dims first.

**Scale boundary:** With a larger baseline corpus (n≥500), PCA or tSNE could identify a more principled set of informative dims. For n=50 baseline, variance-sort is the practical choice.

---

## Leading vs Lagging Indicators

**Decision:** Embedding drift signals (KS, centroid cosine) are primary/leading. LLM quality scores are secondary/lagging. Both are monitored; alerting prioritizes the leading indicators.

**Why:** The LLM handles short OOD questions adequately even without retrieved context — quality scores degrade slowly. Embedding centroid drift and retrieval similarity move earlier because they directly measure semantic distance from the training distribution. Alerting on leading indicators gives more response time.

**Tradeoff:** Leading indicators have false-positive risk — a batch of unusual-but-valid ML questions may trigger embedding drift without quality degradation. The design accepts this: false positives are preferable to late detection.

---

## LLM Judge Alerts Disabled at n=5

**Decision:** `check_thresholds()` does not fire alerts on `avg_quality_score` or `hallucination_rate`. Both are logged to MLflow and W&B for trend visualization.

**Why:** At n=5 queries per batch, each hallucination flag shifts the rate by 20%. A single outlier response produces `hallucination_rate = 0.20` — indistinguishable from genuine degradation. Hard alerts at n=5 would fire constantly on noise.

**Tradeoff:** Genuine quality degradation in the LLM output goes undetected unless retrieval similarity also drops. The assumption is that retrieval similarity is a reliable proxy — if the model lacks relevant context, quality will degrade. This is a documented assumption, not a guarantee.

**Scale boundary:** At n≥15 per batch, hallucination rate variance drops enough (each flag = 6.7% shift) to use as an alert signal. Wire `check_thresholds()` to generate `quality_degraded` and `hallucination_spike` keys when `sample_size >= 15`.

---

## Retrieval Corpus: Inter-Repo Dependency

**Decision:** `quality.py` uses `lora-finetune/data/train.jsonl` as the retrieval corpus via a relative path three levels up in the monorepo.

**Why:** The lora-finetune training data is StackOverflow ML Q&A — the same distribution as the simulated batch queries. Using it as a retrieval corpus means in-distribution queries score high similarity and OOD queries score low, which is exactly the signal we need.

**Tradeoff:** Path resolves only in the monorepo layout. Cloning `llm-drift-monitor` independently breaks `CORPUS_PATH.exists()` — `_load_corpus()` silently returns empty list → all retrieval scores are 0.0 with no diagnostic.

**Fix (planned):** Log a startup warning if `CORPUS_PATH.exists()` is False.

---

## SLO Framework for LLM Monitoring Services

Applying the SLI/SLO/SLA pattern to this drift detection pipeline explicitly — not just documenting scale limits but defining what "working correctly" means and when to page.

### Service Level Indicators (SLIs)

| Signal | Indicator | Type |
|---|---|---|
| Embedding drift | KS statistic on top-20 dims per batch | Leading |
| Semantic drift | Cosine distance of batch centroid from baseline | Leading |
| Retrieval quality | Mean retrieval similarity score per batch | Leading |
| Output quality | Average LLM judge score (n≥15 per batch) | Lagging |
| Hallucination rate | Fraction of responses flagged per batch (n≥15) | Lagging |
| Monitor latency | Time from batch arrival to alert emission | Operational |

### Service Level Objectives (SLOs)

| SLI | Target | Window |
|---|---|---|
| KS stat (embedding) | < 0.3 at p < 0.05 for 95% of batches | Rolling 7-day |
| Centroid cosine dist | < 0.3 for 95% of batches | Rolling 7-day |
| Retrieval similarity | ≥ 0.35 mean for 90% of batches | Rolling 7-day |
| Monitor latency | Alert emitted within 2 batches of threshold breach | Per-breach |
| MLflow write success | 100% of completed batches logged | Cumulative |

### Burn Rate Alert Design

At hourly batch cadence, the 7-day error budget = 168 batches × 5% = 8.4 allowed failures.

- **Fast burn alert (2x rate):** If 4+ failures occur in 1 hour (consuming 50% of budget), page immediately
- **Slow burn alert (sustained):** If 6+ failures occur in 24 hours, escalate to on-call review
- **Leading indicator advantage:** Embedding drift fires 2–3 batches before retrieval quality drops — creates a buffer to investigate before SLO breach

### Golden Signals for This Pipeline

| Signal | ML Translation |
|---|---|
| Latency | Time from `run_batch()` call to `check_thresholds()` completion |
| Traffic | Batch arrival rate (batches/hour); sample size per batch (n) |
| Errors | MLflow write failures, W&B API errors, embedding model unavailable |
| Saturation | Baseline array size vs. available RAM; embedding model queue depth |

### What This Framework Enables at Org Scale

This SLO design is a reusable template. Any team shipping an LLM pipeline can adopt the same SLI taxonomy (leading embedding signals, lagging quality signals) and calibrate their own thresholds from observed in-distribution data. The MLflow/W&B ownership split — audit vs. live alerting — is the architectural decision that makes this framework operationally durable rather than a one-off dashboard.

---

## What Was Cut

| Cut | Reason | Upgrade trigger |
|---|---|---|
| Prometheus + Grafana | Setup overhead disproportionate to demo | Production deployment at scale |
| Evently AI drift reports | Adds third tool, less time-series capability | Rich drift HTML reports for stakeholders |
| Async batch processing | Single-threaded is simpler for demo | >10 concurrent batch streams |
| Model re-indexing on drift | Out of scope: this project monitors, not remediates | Automated remediation pipeline |
| Hard LLM judge alerts | Too noisy at n=5 | Increase sample_size to 15+ |
| Token counting on batch sizes | No budget management needed for demo | Production context window management |
