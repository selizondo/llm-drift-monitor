# ADR-01: Tool Split — MLflow vs Weights & Biases

**Status:** Accepted  
**Date:** 2025-05-09  
**Context:** LLM Production Monitoring System (Project 08)

---

## Context

The monitoring pipeline needs two capabilities that are often served by one tool:

1. **Audit trail** — a permanent, queryable record of what happened per batch: which metrics, which alerts fired, what the drift report contained, and what was done in response.
2. **Live dashboard** — a time-series view of how metrics trend across batches, with visual alert thresholds and the ability to see degradation in progress.

Both MLflow and W&B can do both. The question is whether to use one or two.

---

## Decision

Use **both tools**, with a hard ownership boundary:

| Tool | Owns | Answers the question |
|---|---|---|
| **MLflow** | Per-batch run records, drift report artifacts, baseline snapshot, alert tags | "What happened in batch 6, and what triggered the alert?" |
| **W&B** | Time-series visualization, cross-batch trend lines, alert threshold lines | "What is happening now, and is the system degrading?" |

The boundary is enforced in code: `monitor/logger.py` handles all MLflow writes; `monitor/dashboard.py` handles all W&B writes. Neither file imports from the other.

---

## Why Not One Tool?

**MLflow alone:**  
MLflow's UI is designed around runs and experiments, not time-series dashboards. Plotting "drift score over batch number" requires exporting data and building charts manually. The artifact model is excellent for audit; the visualization layer is not built for live trend monitoring.

**W&B alone:**  
W&B's run history is not designed to serve as a durable audit log. You can query it, but it's optimized for visualization, not structured querying ("show me all batches where alert_types contained 'embedding_drift'"). The Model Registry in W&B exists but is less mature than MLflow's for versioning index artifacts.

**Both:**  
Each tool does what it's good at. Cost is low: both have free tiers, and the code overhead is two thin modules (~30 lines each). The real cost is conceptual: you need to know which tool to open for which question. This ADR is that documentation.

---

## Ownership Details

**MLflow owns:**
- `baseline` run: stores `baseline_stats.json` (embedding distribution at training time)
- `batch-NN` runs: per-batch metrics as MLflow metrics, drift report JSON as artifact, alert types as tags
- Future: model/index versioning in the MLflow Model Registry if re-indexing is triggered

**W&B owns:**
- `monitoring-session` run: time-series of all drift and quality metrics with batch as x-axis
- `drift/*`, `quality/*`, `alerts/*` metric namespaces for clean dashboard organization
- Alert threshold lines set manually in the W&B UI against the logged 0/1 alert flags

---

## Consequences

**Positive:**
- Audit trail is queryable via MLflow CLI/API independent of W&B account status
- Dashboard is visual and shareable without MLflow being running
- Clear separation means each tool is maintainable without understanding the other

**Negative:**
- Two accounts, two API keys, two places to look
- A run that produces no alerts in W&B might still have useful detail in MLflow — requires knowing to check both
- If W&B changes pricing or API, the dashboard layer needs replacing; audit layer is unaffected (and vice versa)

---

## Alternatives Considered

**Evidently AI:** Purpose-built for data drift with a rich HTML report format. Rejected because it doesn't provide the same time-series dashboard capability and adds a third tool. Would be the right choice for a richer drift report in a production system.

**Single MLflow instance with custom plots:** Feasible but requires significant custom charting code. The W&B dashboard is more immediately readable to non-technical stakeholders.

**Prometheus + Grafana:** The right architecture for a real production system at scale. Rejected for this project because setup overhead (cluster, exporters, PromQL) is disproportionate to the demo. Noted in README as the production path.
