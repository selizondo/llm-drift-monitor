# Product and Cost

This document frames the project for a technical business reviewer: what organizational risk it addresses, how the system earns trust, what it costs to operate, and when a team should build something like this versus buying a dedicated drift monitoring service.

---

## The Business Problem

Most AI teams monitor accuracy. Accuracy is a lagging indicator: you measure it after answers are produced, often after users have already experienced degraded quality. By the time accuracy drops visibly, the distribution shift that caused it may have been present for days or weeks.

The organizational risk is not that LLM systems degrade: all systems degrade. The risk is that degradation is invisible until it surfaces as a customer complaint, a missed SLA, or an unexplained drop in engagement metrics. At that point, diagnosing the root cause requires reconstructing what changed in the input distribution, when it changed, and whether the model was the cause or a symptom.

This monitor builds the observability layer before an incident, not after. The audit trail in MLflow means you can answer "when did batch 6 start drifting?" without reconstructing from logs.

---

## Trust Surface

**What can go wrong in an unmonitored LLM system:**

- Input distribution shifts (users ask different types of questions) and the model's retrieval context becomes progressively less relevant, degrading answer quality over weeks with no alert
- A corpus update or model swap changes the embedding distribution, causing retrieval to silently degrade before any quality metric flags it
- OOD inputs (off-topic queries, domain shift from a new customer segment) reach the model without retrieval context, producing confident but unsupported answers

**How this monitor addresses each:**

- Embedding drift (KS test on top-20 variance dimensions) fires 2 batches before quality scores drop. The team can investigate before users are affected.
- Centroid distance tracks directional semantic shift: a gradual drift in query topic shows up as a slow centroid migration before it reaches the KS threshold.
- Quality scores use a self-judging LLM at temperature 0 on a 1 to 3 scale with a documented variance model. Alerts are disabled below n=15 samples because at n=5, a single hallucination flag shifts the rate by 20% (noise, not signal).
- MLflow provides an immutable audit trail per batch: metrics, drift report artifact, alert tags. W&B provides the live time-series view. The ownership boundary is documented so the team knows which tool to query for which question.

**What is not addressed here:** This monitor runs batch-by-batch on a simulated stream, not on live production traffic. For real-time production monitoring, replace the batch files with a Kafka consumer. For per-query confidence scoring, add a confidence estimator to the quality module. For automated remediation (re-index on drift), wire the MLflow alert tag to a downstream job.

---

## Cost Model

The monitor itself has minimal running cost. The variable cost is the LLM quality scoring.

| Component | Cost | Notes |
|-----------|------|-------|
| MLflow | $0 | Open source, local or remote tracking server |
| W&B | $0 | Free tier sufficient for 1 project, 100 GB storage |
| Embedding model | $0 | all-MiniLM-L6-v2 runs locally on CPU |
| LLM quality scoring | ~$0.002 to $0.005 per batch (n=5) | Claude Haiku at ~$0.80/1M input tokens |
| LLM quality scoring | ~$0.005 to $0.015 per batch (n=15) | Recommended for production alerting |

**At hourly batch cadence:**
- n=5 quality sample: ~$0.12 to $0.36/day, ~$3.60 to $10.80/month
- n=15 quality sample: ~$0.36 to $1.08/day, ~$10.80 to $32.40/month
- `--no-quality` flag: $0 (drift detection only, no LLM calls)

**Comparison:** Dedicated drift monitoring platforms (Arize, Evidently Cloud, WhyLabs) typically cost $500 to $2,000/month at small scale. The custom monitor covers the core use case (embedding drift, quality monitoring, audit trail) at under $35/month.

**Inflection point:** At high batch volume (1,000+ batches/day), MLflow local file-based tracking becomes a write bottleneck. Migrate to a remote tracking server (Postgres backend). W&B handles volume natively.

---

## Market Context

The LLM observability market grew rapidly in 2023 to 2025 alongside LLM adoption. Dedicated platforms (Arize Phoenix, Evidently AI, WhyLabs, Langfuse) emerged to fill the gap. Most are optimized for one of two use cases: trace-level logging (individual query/response pairs) or dataset-level drift reports (batch comparison).

The gap they share: the hand-off between the two. A team that detects embedding drift wants to know which specific batches caused it, which alert fired first, and what the retrieval quality looked like in those batches. That cross-cutting query requires both a time-series view and a structured audit trail.

This project demonstrates the pattern with two tools rather than one: MLflow for the audit trail, W&B for the time-series. The split is the architectural insight, not the tools themselves. Teams on Prometheus + Grafana would implement the same pattern with different tooling.

---

## Deployment Constraints

**Batch vs real-time:** This monitor processes queries in fixed batches (50 queries per batch in the demo). For production, replace `data/simulate_stream.py` with a Kafka consumer. The monitoring modules (embeddings, drift, quality) are batch-oriented by design and work unchanged with any source that produces a list of queries.

**Corpus path dependency:** `quality.py` resolves the retrieval corpus relative to the monorepo layout (`../../../lora-finetune/data/train.jsonl`). Independent deployment requires setting `CORPUS_PATH` to an absolute path. A warning is emitted if the corpus is not found; the monitor continues with retrieval scores of 0.0.

**Latency per batch:** At n=5 quality samples and 50 queries per batch, a full monitoring pass completes in approximately 30 to 60 seconds (embedding + drift + quality + logging). At n=15, approximately 60 to 120 seconds. Both are suitable for hourly batch cadence; neither is suitable for per-query real-time monitoring.

**On-call implications:** Alert types are logged as MLflow tags and W&B metrics. A batch with `embedding_drift` in its MLflow tags warrants investigation of the input distribution for that batch window. No automated remediation: the monitor detects and records, the team decides and acts.

---

## Build vs Buy

**Build (this approach) when:**

- The team needs drift detection as part of a custom pipeline where managed platforms do not have a native integration
- Cost is a constraint and the team can operate with $35/month versus $500 to $2,000/month for a managed platform
- The team wants to own the SLO definitions and threshold calibration rather than using platform defaults
- The observability story for a portfolio or internal tool needs to be auditable and reproducible without a vendor dependency

**Buy or use a managed platform (Arize, Evidently, WhyLabs, Langfuse) when:**

- The team needs rich out-of-the-box drift reports with HTML visualizations for non-technical stakeholders
- Real-time per-query observability is required (managed platforms handle streaming natively)
- The team's pipeline is standard enough (LangChain, OpenAI) that native integrations exist
- Scale exceeds 10,000 batches/day and the team does not want to operate a remote MLflow server

**The judgment call:** The custom monitor demonstrates the architectural pattern that managed platforms implement: leading indicators (embedding drift) separate from lagging indicators (quality scores), with an audit trail separate from a live dashboard. Teams that understand the pattern can evaluate managed platforms on their own merits rather than adopting them because drift monitoring feels complex. That understanding is worth building before committing to a vendor.
