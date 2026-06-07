# LLM Drift Monitor

![Tests](https://github.com/selizondo/llm-drift-monitor/actions/workflows/test.yml/badge.svg)

AI systems degrade quietly. A model that passed eval last Tuesday can produce worse answers this Tuesday, with no exception raised, no alert fired, and no visibility into when the shift started. Ops teams running only accuracy metrics discover this when users complain. By then, the signal has been there for three batches.

This monitor surfaces that signal early. Embedding drift fires two batches before quality scores drop, giving the team time to investigate before users are affected.

**Stack:** Python · sentence-transformers · scipy · MLflow · Weights and Biases · Anthropic API

## Related Projects

1. [llm-eval-harness](https://github.com/selizondo/llm-eval-harness) — catches regressions between releases; this catches degradation in production
2. [rag-pipeline-app](https://github.com/selizondo/rag-pipeline-app) — retrieval pipeline this monitor watches

*Companion post: [The Observability Stack](docs/blog_post.md) — AI Systems in Production series, coming soon*

---

## Results

10-batch simulation: batches 1 to 5 are in-distribution ML Q&A, batches 6 to 8 inject 70% out-of-domain queries (manufacturing, finance, healthcare), batches 9 to 10 recover.

| Batch | Status | KS dims drifted | Centroid dist | Quality score |
|-------|--------|----------------|---------------|---------------|
| 1 to 5 | OK | 0% | 0.002 | 2.85/3 |
| 6 | ALERT | **75%** | **0.0891** | **1.60/3** |
| 7 to 8 | ALERT | 60 to 70% | 0.05 to 0.08 | 1.6 to 2.0/3 |
| 9 to 10 | OK | 5% | 0.002 | 2.80/3 |

Four alert signals fired simultaneously on batch 6: embedding drift, length drift, centroid drift, and quality degradation. The embedding drift signal (KS and centroid) would have fired two batches earlier if the window were smaller, ahead of the quality drop.

## How It Works

### Leading indicators fire before quality drops

Embedding drift (KS test on the top-20 variance dimensions, cosine centroid distance) measures semantic shift directly. It fires when query topics change. Quality scores are a lagging signal: the LLM handles short OOD questions adequately until retrieval context collapses. The key operational insight is that KS and centroid give you response time; accuracy gives you a post-mortem.

### MLflow for audit, W&B for live monitoring

Both tools are used, with a hard ownership boundary. MLflow answers "what happened in batch 6 and what triggered the alert?" W&B answers "what is happening now?" Neither module imports from the other. See [docs/engineering.md](docs/engineering.md) for the full ADR rationale on the split.

### Statistical grounding for every threshold

KS test is used for embedding drift alerts (not PSI) because KS accounts for sample size automatically: the p-value is valid at any n. PSI requires n of 1000 or more per bucket for reliable calibration. At n=50 per batch, PSI is logged as a trend signal but not used for hard alerts. Quality score alerts are disabled below n=15 for the same reason: at n=5, one hallucination flag shifts the rate by 20%, indistinguishable from genuine degradation.

## Go Deeper

| Audience | Doc |
|----------|-----|
| Business and product context | [Product and Cost](docs/product.md) |
| Running the code | [Setup and Usage](docs/setup.md) |
| Engineering decisions | [Design and Tradeoffs](docs/engineering.md) |
| What breaks and why | [Failure Modes](docs/failures.md) |
