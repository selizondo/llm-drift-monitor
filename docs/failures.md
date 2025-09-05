# Failure Scenarios

Failure modes for the monitor. "Handled" means a try/except with a non-fatal path exists. "Documented gap" means the failure is understood but detection is not yet implemented.

---

## Failure 1: Corpus Path Missing (Documented Gap)

**What breaks:** `quality.py` resolves `CORPUS_PATH` to `../../../lora-finetune/data/train.jsonl`. If the path doesn't exist (independent clone, corpus moved), `_load_corpus()` silently returns an empty list. `retrieval_quality_score()` returns `avg_retrieval_sim = 0.0` and `retrieval_miss_rate = 0.0` — indistinguishable from all queries being OOD.

**Status:** Documented gap — no startup check.

**Detection (planned):** Add to `_load_corpus()`:
```python
if not CORPUS_PATH.exists():
    print(f"[quality] WARNING: corpus not found at {CORPUS_PATH}. Retrieval scores will be 0.", file=sys.stderr)
```

---

## Failure 2: Anthropic API Unavailable (Partially Handled)

**What breaks:** `score_batch_sample()` calls `client.messages.create()` for answer generation and judging. If the API is down, rate-limited, or the key is invalid, all judge calls fail.

**Status:** Partially handled — broad `except Exception` catches API errors, logs to stderr, appends neutral score (2.0). Monitor continues running; `--no-quality` flag skips LLM scoring entirely for API-free operation.

**Observable:** Stderr log line: `[quality] judge call failed for query '...': <error>`. Quality metrics will cluster at 2.0 for the affected batch — visible in W&B trend as a flat line, not a genuine quality signal.

**Remaining gap:** No distinction between transient API error (retry) and hard failure (invalid key). A batch with all API failures silently produces `avg_quality_score = 2.0`.

---

## Failure 3: Embedding Model Download Fails (Documented Gap)

**What breaks:** `embeddings.py` calls `SentenceTransformer(MODEL_NAME)` at first use. If HuggingFace is unavailable and the model isn't cached locally, the monitor crashes with an unhandled connection error.

**Status:** Documented gap — no retry or fallback.

**Detection (planned):** Wrap model load in try/except with a clear message:
```python
try:
    _model = SentenceTransformer(MODEL_NAME)
except Exception as e:
    raise RuntimeError(
        f"Failed to load embedding model '{MODEL_NAME}'. "
        f"Check HuggingFace connectivity or pre-cache the model. Original error: {e}"
    )
```

---

## Failure 4: W&B Init Fails (Documented Gap)

**What breaks:** `dashboard.init_run()` calls `wandb.init()`. If the W&B API key is missing or invalid, `wandb.init()` raises. The monitoring loop hasn't started yet, so all batches fail.

**Status:** Documented gap — no fallback to MLflow-only mode.

**Detection (planned):** Wrap `wandb.init()` in try/except; if it fails, set a `_wandb_available = False` flag and skip W&B logging for the session. MLflow continues as the audit trail. `run_monitor.py` already has `--no-wandb` for this use case — surface the W&B error as a warning and suggest the flag.

---

## Failure 5: MLflow Tracking Server Unreachable (Documented Gap)

**What breaks:** If using a remote MLflow tracking server (set via `MLFLOW_TRACKING_URI`), `mlflow.start_run()` will fail if the server is unreachable. Local file-based MLflow (default) is not affected.

**Status:** Documented gap — not applicable to local demo setup. Relevant for production deployments.

**Detection (planned):** Check `mlflow.get_tracking_uri()` at startup; if remote, test connectivity before starting the monitoring loop.

---

## Failure 6: All Embedding Dims Have Zero Variance (Documented Gap)

**What breaks:** `_select_informative_dims()` sorts by variance and picks the top N. If all dims have zero variance (e.g., empty baseline batch, all identical queries), `np.argsort` returns valid indices but the selected dims carry no signal. All KS tests will return p=1.0 (no drift detected regardless of actual distribution).

**Status:** Documented gap — edge case, unlikely in practice.

**Detection (planned):** Check `baseline_embeddings.var(axis=0).max() > 0` before running drift detection; raise a warning if baseline has degenerate variance.
