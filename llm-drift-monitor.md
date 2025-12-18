# **STAFF REVIEW: LLM Drift Monitor**

## Executive Summary

The llm-drift-monitor repo is a **well-architected monitoring layer for LLM production systems** that detects input, embedding, and output distribution drift before users notice degradation. The design demonstrates strong system thinking (contract-first, observability-first, two-tool architecture), but has **critical production gaps in error handling and insufficient test coverage** that block deployment.

**Architecture Grade: A−**  
**Production-Readiness Grade: C+ (fixable in 2–3 days)**

---

## Architecture & System Design

### ✅ **Contract-First Design — STRONG**

The system defines clear contracts upfront:

- **Single embedding model source of truth:** `MODEL_NAME = "all-MiniLM-L6-v2"` in [embeddings.py](monitor/embeddings.py#L6) is the only place it's defined; no version strings scattered across configs.
- **Baseline captured and versioned:** `compute_baseline()` snapshots the distribution once (batch 1), then all subsequent batches measure against it. The baseline is persisted in MLflow as an artifact.
- **Unified threshold definitions:** All alert thresholds live in one file ([trends.py](monitor/trends.py#L1-L22)) with inline documentation of expected ranges.
- **Ownership boundary enforced in code:** MLflow writes in [logger.py](monitor/logger.py); W&B writes in [dashboard.py](monitor/dashboard.py). Neither imports the other. This is the best kind of contract — it's structural.

**Why this matters:** Every other project in your portfolio has some version of this pattern. It prevents the "change one thing, forget to update the other three places" bug.

---

### ✅ **Observability as Response Fields — STRONG**

Degradation signals are surfaced in structured outputs, not buried in logs:

| Signal | Location | Format |
|--------|----------|--------|
| Drift metrics | MLflow metrics + W&B payload | JSON scalars with batch#, p-values, percentages |
| Alert tags | MLflow tags | `alert_types=embedding_drift,length_drift` for filtering |
| Drift report | MLflow artifact | Full KS details per dimension |
| Dashboard namespace | W&B | `drift/pct_dims_drifted`, `quality/avg_retrieval_sim`, `alerts/embedding_drift` |

The [summarize_batch()](monitor/trends.py#L50-L63) output is a model of this:
```
Batch 06 [ALERT]
  drift  : 75% dims | len_p=0.001 | centroid=0.0891
  quality: retrieval_sim=0.28 | miss=15%
  ! embedding_drift: pct_dims_drifted=75% (threshold 30%)
```

Every field a caller needs to decide "should I act?" is there — no log parsing required.

---

### ⚠️ **Non-Fatal Degradation — PARTIAL (Gap)**

The system *attempts* graceful degradation but has **critical unprotected paths:**

**What works:**
- If corpus is missing: `_load_corpus()` returns `[]`, `retrieval_quality_score()` returns `0.0`, quality alert fires naturally.
- If batch is empty: validated and skipped without crashing.
- If embedding is empty: handled with size checks.

**What breaks (will crash in production):**
1. **Anthropic API failure:**
   ```python
   # monitor/quality.py, line 108
   answer = _call_haiku(client, system, user_msg)  # ← No try/except
   judge_raw = _call_haiku(client, system, judge_prompt)  # ← will raise on network error
   ```
   A rate-limit, timeout, or network blip will crash the entire batch. The scores aren't logged, the drift report isn't saved, monitoring stops.

2. **Embedding model load failure:**
   ```python
   # monitor/embeddings.py, line 16
   _model = SentenceTransformer(MODEL_NAME)  # ← will crash if CUDA unavailable or model missing
   ```
   If the download fails or CUDA isn't available, there's no fallback to CPU. System crashes at the first batch.

3. **MLflow/W&B logging failure:**
   If MLflow server is unreachable or W&B API key is invalid, the batch loop exits. No partial success.

**Recommendation:** Wrap all external I/O in try/except blocks. Return degraded signals (e.g., `quality_score=2.0, hallucinationrate=0.0` if API fails), log the failure to MLflow, continue monitoring.

---

### ✅ **Failure Modes Documented — STRONG**

The README explicitly names and explains failure patterns:

| Failure | Root Cause | Signal |
|---------|-----------|--------|
| **Input drift** | Users ask OOD questions | KS test fires on embedding dims + PSI on length |
| **Embedding drift** | Query semantics change | Centroid cosine drift (leading indicator) |
| **Output quality drift** | Retrieval fails on OOD queries | Retrieval similarity drops, LLM score follows |

**Key insight:** Quality degradation is *structural*, not artificially injected. OOD queries don't match the ML corpus (sim < 0.3) → model answers without grounding → judge scores drop naturally. This is honest failure characterization.

**Also documented:** How the system recovers (batches 9-10 return to in-distribution), proving the detector isn't permanently fooled.

---

### ⚠️ **Baselines Before Improvements — PARTIAL (Gap)**

A baseline is captured, but **no naive comparison.**

- ✅ Baseline from batch 1 stored in MLflow (`n_samples=50`, centroid vector, variance per dimension).
- ✅ All metrics measured relative to baseline (KS test against baseline distribution, centroid distance from baseline mean).
- ❌ **Missing:** What does the baseline look like? What if we used a random baseline (cosine distance from a random vector)? What if we used a majority-class baseline for quality?
- ❌ **Missing:** Expected performance for in-distribution queries. Is 2.85/3 quality good? Good compared to what?

**Recommendation:** Add a simple baseline comparison doc:
```
In-distribution baselines (batches 1-5):
  - centroid_drift: mean=0.008, std=0.003, max=0.019
  - retrieval_sim: mean=0.62, std=0.04
  - quality_score: mean=2.85, std=0.10

OOD baselines (batches 6-8):
  - centroid_drift: mean=0.55, std=0.03
  - quality_score: mean=1.65, std=0.40
```

This gives context to the thresholds.

---

### ⚠️ **Explicit Scale Boundaries — PARTIAL (Gap)**

The system has design limits, but they're not documented.

**What breaks at scale?**

| Component | Boundary | Status |
|-----------|----------|--------|
| Batch size | 50 queries | ✅ Works (50 = fast for demo) |
| Embedding model | all-MiniLM-L6-v2 (384-dim) | ⚠️ No max batch size documented |
| Quality sample | 5 queries per batch | ❌ Acknowledged as too small ("each flag = 20%") |
| Corpus size | ~1000s of docs from lora-finetune | ❌ No limit tested |
| Concurrent monitors | 1 | ❌ MLflow.db is file-based (not thread-safe) |
| Time per batch | ? | ❌ Not profiled |

**From the ADR:**
> "Prometheus + Grafana: The right architecture for a real production system at scale. Rejected for this project because setup overhead is disproportionate to the demo."

This is honest but incomplete. **What is "production scale"?** 100K queries per hour? 1M? The boundary isn't explicit.

**Recommendation:** Add a "Scale Boundaries" section in README:
```
**Tested:** 50 queries/batch, 1000-doc corpus, 10 batches (~2 min runtime)

**When this breaks:**
- Batch size > 1000: embedding model becomes slow (~1s per 100 queries on CPU)
- Corpus > 100K docs: retrieval similarity becomes O(n), too slow for per-query scoring
- Quality sample > 30: Anthropic API costs become uneconomical ($0.30/batch at 30 samples)
- Concurrent monitors > 1: MLflow.db file locking becomes a problem
- Monitoring latency > 5 min: drift signals are stale for real-time systems

**Production path:** Upgrade to Kafka for real-time ingestion, Prometheus for metrics, Evidently AI for drift, Postgres for MLflow backend.
```

---

## Data Quality & Evaluation

### ✅ **Drift Simulation — Honest**

The simulated drift is realistic and well-justified:

- **Batches 1–5, 9–10:** Clean ML Q&A (from lora-finetune val set) — in-distribution.
- **Batches 6–8:** 70% out-of-domain + 30% ML Q&A.
- **OOD sources:** 150 hand-authored questions across manufacturing (PLC, SCADA, vibration analysis), finance (VaR, trading, volatility), healthcare (clinical trials, EHR, survival curves), and supply chain (demand forecasting, 3PL).

**Why this is good:**
- Vocabulary overlap (Python, data, model are in both) but semantic distance.
- Realistic: a system might receive manufacturing questions after launch (domain drift).
- Deterministic (seed=42), reproducible.

**What's missing:**
- No label leakage check. (Not applicable here, but document why.)
- Temporal split is synthetic (batches are ordered but data is iid). Real stream drift might have autocorrelation.
- OOD domains are hand-curated, not derived from actual user queries.

---

### ✅ **Metric Selection — Complementary Signals**

The system uses multiple, non-redundant detectors:

| Metric | Method | When It Fires | Leading/Lagging |
|--------|--------|---------------|-----------------|
| **KS dims drifted** | K-S test on top-20 variance dims | Embedding distribution shifted | Leading |
| **Length drift (PSI)** | PSI on query word count | Query pattern changed | Leading |
| **Centroid drift** | Cosine distance from baseline mean | Directional semantic shift | Leading |
| **Retrieval sim** | Avg cosine sim to corpus | Retrieval fails on new queries | Leading |
| **Quality score** | LLM self-judge 1-3 | Answers without grounding | Lagging |

**Why this matters:** Different signals catch different failure modes. A model that drifts orthogonal to the top-20 dims might not fire KS; but centroid drift catches it. A retrieval miss fires immediately; quality score confirms 2-3 batches later.

**The problem:** Quality signal is **undersampled.**
```python
# trends.py: n=5 samples per batch
# quality.py, line 136: "each flag = 20%"
sample_size: int = 5  # ← At n=5: hallucination_rate 0%→100% with 5 queries
```

With only 5 samples, a hallucination rate flips 20% per query. The README acknowledges this: *"use for qualitative trends, not hard alerts."* But the system logs it as if it's quantitative.

**Recommendation:** Increase to `n=15` (cost: 3x more API calls) or lower the alert threshold for quality signals.

---

### ✅ **Ground Truth Quality — PARTIAL (Gap)**

The LLM judge is calibrated informally:

- Uses `claude-haiku-4-5` for quality and grounding assessment.
- Self-judge prompt is well-structured (quality: 1-3, grounded: 1-3, hallucination: bool).
- But: **No comparison to a second judge model.** If you run the same queries with GPT-4o as judge, do you get the same hallucination flags? Probably not.
- Also: Absolute RAGAS numbers (0.358 for dense, 0.625 for BM25) shift with stronger judges — this system reports the numbers without the caveat.

**Recommendation:** Document that scores are relative deltas, not ground-truth absolute scores.

---

### ✅ **Reproducibility — Strong**

- ✅ All experiments deterministic (seed=42).
- ✅ MLflow persists all metrics and drift reports as JSON artifacts.
- ✅ Batches saved to disk (JSONL format, queryable).
- ❌ W&B run history is not designed as an audit log (acknowledged in [ADR-01](docs/adr-01-tool-split.md)).

**Why the two-tool split makes sense:** MLflow is durable (queryable later, not subject to W&B account changes), W&B is visual (better dashboards). The ADR documents this tradeoff explicitly.

---

## Production-Readiness Gaps

### 🔴 **CRITICAL: API Error Handling Missing**

**Location:** [monitor/quality.py](monitor/quality.py#L100-L110)

```python
def score_batch_sample(...):
    for query in sampled:
        try:
            answer = _call_haiku(...)  # ← No retry, no fallback
            judge_raw = _call_haiku(...)  # ← Will crash on rate limit
            ...
        except Exception:
            quality_scores.append(2.0)  # ← Catches, but doesn't log
            halluc_flags.append(0)
```

**Problem:** If Anthropic API is rate-limited (429), times out (timeout), or has network error, `_call_haiku()` raises. The exception is caught in the inner loop, but only for a single query. The first query failure will propagate out of `score_batch_sample()`, crashing the entire batch.

**Fix:**
```python
def _call_haiku(client, system, user, max_retries=3, backoff=2):
    for attempt in range(max_retries):
        try:
            resp = client.messages.create(...)
            return resp.content[0].text.strip()
        except anthropic.RateLimitError:
            if attempt < max_retries - 1:
                time.sleep(backoff ** attempt)
            else:
                mlflow.log_text(f"API rate limit after {max_retries} retries", "api_error.txt")
                return None  # ← Caller handles None
        except Exception as e:
            mlflow.log_text(f"API error: {e}", "api_error.txt")
            return None
```

### 🔴 **CRITICAL: Corpus Versioning Missing**

**Location:** Entire codebase

**Problem:** If the corpus (lora-finetune/data/train.jsonl) is updated between batch runs, the system has no way to know. Retrieval scores might drop not because of drift, but because the corpus changed. Conversely, if the model's understanding of queries improves, drift might be misattributed to the query distribution.

**Fix:** Store corpus checksum in baseline MLflow run:
```python
def log_baseline(baseline_stats):
    corpus_sha = hashlib.sha256(open(CORPUS_PATH, 'rb').read()).hexdigest()
    mlflow.log_params({
        "corpus_version": corpus_sha[:8],
        "corpus_size": len(_load_corpus()),
    })
```

Then warn if corpus changes:
```python
def validate_corpus_unchanged(baseline_corpus_sha):
    current_sha = ...
    if current_sha != baseline_corpus_sha:
        raise ValueError(f"Corpus changed! {baseline_corpus_sha} → {current_sha}")
```

### 🔴 **CRITICAL: No Unit Tests**

**Location:** No test file exists.

The repo has 15 Python files but zero tests. For a monitoring system, this is critical:

**Minimum tests needed:**
1. `test_drift_detection_fires_on_ood()` — inject OOD batch, verify embedding_drift alert fires.
2. `test_quality_degradation_on_retrieval_miss()` — inject queries that don't match corpus, verify quality drops.
3. `test_graceful_degradation_missing_corpus()` — corpus path doesn't exist, verify no crash and alert fires.
4. `test_baseline_correctness()` — baseline equals batch 1 mean and variance.
5. `test_threshold_tuning()` — verify thresholds from README match code.

**Recommendation:** Add a `tests/` directory with at least the happy path + one failure scenario.

---

### ⚠️ **MEDIUM: Embedding Model Error Handling Missing**

**Location:** [monitor/embeddings.py](monitor/embeddings.py#L14-L20)

```python
def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)  # ← Will crash if CUDA unavailable
    return _model
```

**Problem:** If CUDA is not available or the model download fails, the system crashes. No fallback to CPU, no error message, no clear guidance.

**Fix:**
```python
def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        try:
            _model = SentenceTransformer(MODEL_NAME, device='cuda')
        except torch.cuda.CudaError:
            print("CUDA unavailable, falling back to CPU (slower)")
            _model = SentenceTransformer(MODEL_NAME, device='cpu')
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding model: {e}. Check internet connection and disk space.")
    return _model
```

---

### ⚠️ **MEDIUM: Thresholds Not Justified**

**Location:** [monitor/trends.py](monitor/trends.py#L5-L22)

```python
THRESHOLDS = {
    "pct_dims_drifted": 0.15,        # ← Why 15% and not 20%?
    "ks_length_p_value": 0.05,       # ← Standard 5% p-value
    "centroid_drift": 0.30,           # ← Why 0.30?
    "avg_retrieval_sim": 0.35,        # ← Why 0.35?
    "avg_quality_score": 2.0,
    "hallucination_rate": 0.30,
}
```

The code comments say "tuned from observed data" but doesn't show the distribution. A production system needs sensitivity analysis:

- If threshold is 0.15, what's the false positive rate on in-distribution batches?
- What's the false negative rate (misses real drift)?
- How do thresholds scale with corpus size?

**Fix:** Add a calibration script that shows:
```
Threshold Sensitivity (from simulated data):
  pct_dims_drifted=0.15:
    - in-distribution: 0% false positives (max 5%)
    - OOD batches: 100% catch rate (min 6-8)
  
  centroid_drift=0.30:
    - in-distribution: mean=0.008, max=0.020 (OK)
    - OOD: mean=0.55 (OK)
```

---

### ⚠️ **MEDIUM: Silent Corpus Loading Failure**

**Location:** [monitor/quality.py](monitor/quality.py#L52-L65)

```python
def _load_corpus() -> list[dict]:
    global _corpus_cache
    if _corpus_cache is None:
        cases = []
        if CORPUS_PATH.exists():
            ...
        _corpus_cache = cases  # ← Returns [] if file doesn't exist
    return _corpus_cache
```

If `lora-finetune/data/train.jsonl` doesn't exist (wrong path, deleted, etc.), the system silently loads an empty corpus. Retrieval scores go to 0, a quality alert fires, but the user has no signal that the corpus is missing.

**Fix:**
```python
def _load_corpus() -> list[dict]:
    if not CORPUS_PATH.exists():
        raise FileNotFoundError(
            f"Corpus not found: {CORPUS_PATH}\n"
            f"Expected to find lora-finetune/data/train.jsonl\n"
            f"Run `python ../lora-finetune/data/generate_data.py` or check your workspace structure."
        )
    ...
```

---

## Good Patterns Worth Emulating

### ✅ **Two-Tool Architecture**

The ADR-01 decision to use both MLflow and W&B is a Staff-level insight:

- **MLflow** = "what happened and when" (audit trail, queryable, durable)
- **W&B** = "what is happening now" (live dashboard, visual, shareable)

Neither tool does both well. Using both means each module (~30 lines) focuses on one job. The cost is two accounts and two mental models, but the separation is worth it operationally.

**This is the pattern I'd recommend to other teams:** If one tool would require significant custom code (e.g., building a time-series dashboard in MLflow), use a second tool. Document the ownership boundary explicitly (like ADR-01 does).

---

### ✅ **Command-Line Flexibility**

```bash
python run_monitor.py --no-quality --no-wandb  # Fast iteration, no API calls
python run_monitor.py --batches 10 --quality-sample 3  # Custom scale
```

This allows the dev loop: iterate fast (no external deps), then full run before committing. Good UX.

---

### ✅ **Transparent Degradation**

The README explicitly states when things are too noisy:
> "Note: n=5 per batch produces noisy hallucination rates (each flag = 20%); increase sample_size to 15+ for reliable signal in production."

This is honest engineering. Most systems hide their limitations.

---

## LLM/RAG-Specific Patterns

### ✅ **Faithfulness vs Relevancy Tradeoff Recognized**

The system measures both:
- **Relevancy:** Retrieval similarity (cosine of query vs corpus)
- **Faithfulness:** LLM judge (is the answer grounded in the context?)

Missing: explicit comparison. If retrieval similarity is 0.6 but quality score is 1.5, what's happening? Model is hallucinating despite good retrieval, or retrieval context is wrong?

**Recommendation:** Correlate signals in post-hoc analysis:
```python
# In MLflow analysis
high_sim_low_quality = [batch for batch in batches
                       if batch['retrieval_sim'] > 0.5 and batch['quality'] < 2.0]
# These batches have high relevancy but low quality → model issue, not retrieval
```

---

### ❌ **Version-Scoped Filtering Not Implemented**

The corpus is static. In a real RAG system, you'd filter by metadata (e.g., "use only docs from corpus v2") *before* ANN ranking. This prevents stale documents from ranking first.

**Not needed for this demo, but flag it for production.**

---

### ✅ **Adaptive Fallback Recognized (But Not Implemented)**

The README mentions fallback thresholds:
> "AdaptiveRetriever with BM25 fallback when dense score < 0.3"

This system doesn't implement it (only dense retrieval), but the idea is documented. Good.

---

## Design Tradeoffs (Clear & Justified)

| Tradeoff | Decision | Rationale | Downside |
|----------|----------|-----------|----------|
| Batch vs per-query | Batch (n=50) | Smoother signals, fewer false alerts | Misses rare individual anomalies |
| MLflow vs W&B | Both | Each solves one problem well | Operational complexity |
| Centroid vs full PCA | Centroid | Simple, interpretable, fast | Blind to orthogonal shifts |
| KS vs PSI for alerting | KS on both | Accounts for sample size | PSI is logged but not acted on |
| 20 dims vs all 384 | 20 dims | High-signal, low-noise | May miss low-variance drifts |
| Optional quality scoring | API calls | Realistic but expensive | Noisy at small sample sizes |

All of these are **documented in code comments**, not hidden. This is what staff-level work looks like.

---

## Recommendations (Prioritized)

### 🔴 **Tier 1: Blocks Production (Fix Before Deploying)**

1. **Wrap Anthropic API calls in retry logic** (30 min)
   - Add exponential backoff, max retries.
   - Log failures to MLflow.
   - Return degraded scores if API fails.

2. **Add corpus version tracking** (30 min)
   - Store corpus SHA in baseline.
   - Warn if corpus changes between runs.
   - Log corpus size in MLflow.

3. **Add integration tests** (1 day)
   - Minimum: drift_fires_on_ood(), quality_drops_on_retrieval_miss(), graceful_degradation_missing_corpus().
   - Run in CI/CD before merge.

---

### 🟡 **Tier 2: Improves Reliability (Do Before First Release)**

4. **Increase quality sample size** from 5 to 15 (1 day)
   - Reduces hallucination variance from 20% to 7%.
   - Update threshold docs.
   - Measure API cost impact.

5. **Add embedding model error handling** (30 min)
   - Detect CUDA unavailability, fallback to CPU.
   - Clear error message on download failure.

6. **Document threshold calibration** (2 hours)
   - Show in/OOD distributions for KS, PSI, centroid.
   - Justify why 15% (not 10% or 20%)?
   - Add false positive rate analysis.

7. **Explicit corpus loading failure** (15 min)
   - Raise FileNotFoundError if corpus not found (don't silently return []).
   - Include setup instructions in error message.

---

### 🔵 **Tier 3: Production Scale (Plan After First Release)**

8. **Add re-index trigger** (1 day)
   - If drift detected for N consecutive batches, re-embed corpus.
   - Log new index version in MLflow.
   - Track which model version served which batch.

9. **Explicit scale boundaries** (2 hours)
   - Document when each component becomes slow/unreliable.
   - Add performance profiling (time per batch, API cost per batch).

10. **Migrate to dedicated drift tool** (1 week)
    - Evidently AI or Arize for production.
    - MLflow remains as audit layer.
    - Prometheus/Grafana for real-time alerting.

---

## Interview Frame

**30 seconds:**
> "I built a production monitoring layer for LLM pipelines that detects input, embedding, and output drift before it affects users. The system uses statistical tests (KS test, PSI, cosine distance) to fire leading indicators 2–3 batches before quality scores drop. It logs to MLflow for audit trails and W&B for live dashboards. The key insight: semantic drift fires early; you get time to act before user-facing impact."

**When asked about architecture:**
> "The two-tool split (MLflow + W&B) is intentional. MLflow owns 'what happened — give me the audit trail.' W&B owns 'what is happening now — draw me the live dashboard.' Each tool does one problem well; trying to do both in one tool requires custom code. The trade-off is two accounts and two mental models, but the separation is worth it operationally."

**When asked about production gaps:**
> "The biggest gap is API error handling. If Anthropic rate-limits the quality scoring, the entire batch fails. That's fixable in 30 minutes with retry logic. The second gap is corpus versioning — if the corpus changes, the system can't distinguish real drift from corpus updates. And I'd add tests; right now there are zero. These three things block production; fixing them is 2–3 days of work."

**When asked about tradeoffs:**
> "I sample 5 queries per batch for quality scoring. This is noisy (each flag = 20% of the rate), but it keeps API costs low. In production, I'd increase to 15 and make the cost-quality tradeoff explicit in the docs. The batch-level aggregation (not per-query) smooths out individual anomalies, which is good for avoiding false alerts but bad for catching rare failures. That's a tradeoff I'd revisit for high-stakes systems (medical, financial)."

---

## Summary Table

| Dimension | Grade | Notes |
|-----------|-------|-------|
| **Architecture** | A− | Contract-first, observability-first, two-tool split well-justified |
| **Data quality** | A− | Realistic drift simulation, multiple complementary signals, honest failure modes |
| **Error handling** | C | Missing: API retry, corpus versioning, embedding model fallback |
| **Testing** | C− | Zero tests (critical gap for a monitoring system) |
| **Documentation** | A | Clear README, ADR-01 justifies design, thresholds explained |
| **Code quality** | A− | Clear modules, thin coupling, good comments, no duplication |
| **LLM-specific** | B+ | Faithfulness vs relevancy recognized, version-scoped filtering not implemented |
| **Production-ready** | C+ | Good design, fixable gaps. 2–3 days of work to deploy. |