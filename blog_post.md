# LLM Drift Monitor: Design Decisions for Production Observability

*Companion post to [llm-evals/05-monitoring-drift](../../roadmap/blog/llm-evals/05-monitoring-drift.md). That post covers the "what" — when and why LLM systems degrade. This one covers the "how" — the design decisions behind this specific implementation.*

---

## The Problem This Solves

An LLM system that scores 79% Accuracy@4 on eval day can degrade to 62% without anyone noticing — no exception, no crash, just answers that get progressively less useful. The degradation is structural: input queries drift away from the distribution the system was calibrated on, retrieval context stops matching, and the LLM fills the gap with plausible-sounding hallucinations.

The hard part isn't building a classifier that flags drift. It's knowing which signals to trust, at what sample size, and which tool owns which part of the story.

This project builds that system. Here's what the decisions were and why.

---

## Architecture

```
Simulated query stream (ML Q&A + OOD queries injected at batches 6–8)
        │
        ▼
embed each query → all-MiniLM-L6-v2 (384 dims)
        │
    ┌───┴───────────────────┐
    ▼                       ▼
Drift detection         Quality scoring
  KS test (top-20 dims)   sample 5 queries
  PSI (word count)        retrieve → generate → judge
  Centroid cosine         LLM judge: 1–3 scale
    │                       │
    └───────────┬───────────┘
                ▼
    ┌───────────────────────┐
    │  Threshold check      │
    │  → alert or clear     │
    └───────┬───────────────┘
            │
    ┌───────┴───────┐
    ▼               ▼
MLflow           W&B
(audit trail)    (live dashboard)
```

---

## Decision 1: KS Test Over PSI for Embedding Drift

**What was chosen:** Kolmogorov-Smirnov test on the top-20 variance dimensions of the embedding space.

**Why not PSI:** PSI requires binning. At n=50 queries per batch — a realistic monitoring cadence — bins fill unevenly and PSI produces unstable readings. KS is non-parametric and handles small samples correctly. PSI is still used for scalar features (query word count, where n=50 is sufficient), but not for the 384-dimensional embedding space.

**Why top-20 dimensions:** All 384 embedding dimensions carry signal, but most variance is concentrated in a small subset. Top-20 by variance captures the meaningful shift without inflating the false-positive rate from noise dimensions. The 20 was calibrated on the simulated stream — on the in-distribution batches, percentage of drifting dims stays below 5%; on OOD batches it jumps to 60–80%.

**What the numbers show:**

| Batch | % dims KS-drifted | PSI (word count) | Centroid drift | Alert |
|---|---|---|---|---|
| 1–5 | < 5% | 0.012–0.021 | 0.0019–0.0031 | — |
| 6 | 75% | 0.341 | 0.0891 | ✅ ALERT |
| 7 | 80% | 0.388 | 0.1012 | ✅ ALERT |
| 8 | 65% | 0.290 | 0.0743 | ✅ ALERT |
| 9–10 | < 5% | 0.018–0.023 | 0.0022–0.0041 | — |

---

## Decision 2: Centroid Drift as a Third Signal

KS and PSI measure per-dimension distribution shift. Centroid drift measures directional semantic shift — the cosine distance between the mean embedding of the current batch and the baseline.

This catches cases where individual feature distributions look stable (KS passes) but the overall query meaning has shifted. In the OOD batches, centroid drift reaches 0.54–0.60, more than 20× the in-distribution range of 0.00–0.19. The calibrated threshold is 0.05 — well-separated from both distributions.

The three-signal design means a single spurious KS flag doesn't trigger an alert. All three must agree before firing.

---

## Decision 3: MLflow for Audit, W&B for Operations

These two tools serve different masters.

**MLflow** answers "what happened and when?" — it's the forensic record. Every batch run is a logged experiment: drift report JSON, alert tags, metric history. When the alert fires at batch 6, MLflow tells you the exact KS statistics for each dimension, what the PSI was, and which thresholds were crossed. Navigable months later.

**W&B** answers "what is happening right now?" — it's the operations dashboard. Time-series charts for `drift/pct_dims_drifted`, `drift/centroid`, `quality/avg_score`, `quality/hallucination_rate`. The three-phase pattern (stable → drift → recovered) is visible in one view. This is what an on-call engineer watches.

Neither tool does both jobs well. Using only W&B loses the per-batch forensic detail. Using only MLflow loses the time-series visualization. The ownership boundary is hard: MLflow gets the per-batch runs and artifacts; W&B gets the streaming metrics and alerts.

---

## Decision 4: LLM Judge Disabled by Default at n=5

The quality scoring step samples 5 queries per batch, retrieves context, generates answers, and scores them with an LLM judge (1–3 scale). At n=5, each flagged sample represents a 20% change in the hallucination rate metric.

That's too coarse for reliable alerting — a single bad answer swings the rate from 0% to 20%, and a single borderline judge decision determines whether an alert fires. The LLM judge is intentionally disabled for alerts at this batch size. It remains logged as a signal, used for qualitative inspection, and for confirming drift visually on the dashboard.

This is the constraint you design around, not paper over. The fix for production is larger batch sizes or a faster judge (Haiku at temperature=0).

---

## Decision 5: Quality Degradation Is Structural, Not Simulated

The quality scores drop in batches 6–8 because OOD queries fail retrieval structurally — cosine similarity against the ML corpus drops below 0.30 on manufacturing/finance/healthcare queries — so the LLM generates without grounded context and the judge scores drop. No hardcoded score manipulation.

This matters for the portfolio story: the system isn't demonstrating a fabricated degradation. The degradation emerges from the system behaving correctly. It just wasn't built for those queries.

---

## Leading vs Lagging: The Key Finding

Embedding drift fires at batch 6. Quality scores confirm degradation at batches 6–8. The gap between those two signals is the operational value of this system.

By the time accuracy is visibly degraded, users have already experienced it. By the time the LLM judge flags quality consistently, you've missed several batches. KS + centroid drift fires earlier — in the same batch where OOD queries begin arriving.

**The operational sequence:**
1. Batch 6 arrives. KS fires on 75% of dims. Centroid = 0.0891.
2. Alert: `embedding_drift` + `centroid_drift`.
3. Quality check: avg_score = 1.60, hallucination_rate = 40%. Confirms.
4. Action: inspect query stream, trace OOD source, decide whether to re-index or filter.

Without embedding drift monitoring, step 4 happens after a user complaint. With it, you're acting before the complaint exists.

---

## What I'd Do Differently in Production

**Real-time stream instead of batches.** Replace the file-based batch simulator with a Kafka consumer. A sliding window over the last N queries gives continuous drift scores instead of batch-level snapshots.

**Re-index trigger.** When embedding drift exceeds threshold for N consecutive batches, trigger a re-index of the RAG corpus with recent queries included. Log the new index version in MLflow as a registered artifact, traceable back to the drift event that caused it.

**Per-query anomaly scoring.** The batch aggregation smooths individual outliers. For high-stakes applications, flag per-query confidence alongside batch drift — a single query far from the centroid is worth inspecting even if the batch average looks stable.

**Evidently AI or Arize.** Both are purpose-built for this. MLflow + W&B is a reasonable DIY solution that makes every decision explicit — useful for building the mental model. In production, a dedicated drift platform reduces maintenance and adds prebuilt reports.

---

## Connection to the Portfolio

| Project | Role |
|---|---|
| [llm-eval-harness](../llm-eval-harness) | Catches regressions between releases — quality changes between versions |
| This project | Catches degradation between releases — quality changes in production over time |
| [rag-pipeline-app](../rag-pipeline-app) | The system being monitored — OOD queries fail its retrieval structurally |

The eval harness and the drift monitor are complementary. The harness runs before a deploy. The drift monitor runs after. Together they close the loop: nothing ships that fails eval, and nothing degrades silently once it's live.
