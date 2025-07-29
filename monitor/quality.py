"""
quality.py — Per-batch quality monitoring using retrieval similarity as the primary signal.

Primary signal (always computed, no API calls):
  retrieval_quality_score() — for each query, compute cosine similarity to best match
  in the ML Q&A corpus. In-distribution queries score ~0.5-0.7; OOD queries score ~0.1-0.3.
  This directly measures whether the system has relevant knowledge for incoming queries.

Secondary signal (optional, requires ANTHROPIC_API_KEY):
  score_batch_sample() — sample N queries, generate answers via claude-haiku-4-5,
  self-judge quality (1-3) and hallucination risk. More realistic but noisier at small N.
  Note: n=5 per batch produces noisy hallucination rates (each flag = 20%); increase
  sample_size to 15+ for reliable signal in production.

The retrieval similarity is the leading indicator. LLM quality scores lag because
the model handles short OOD questions adequately even without retrieved context.
"""

import json
import os
import random
import re
from pathlib import Path

import anthropic
import numpy as np

# Use lora-finetune training data as the retrieval corpus.
# It's the same StackOverflow ML Q&A distribution as the batch queries, so
# in-distribution queries score high similarity; OOD queries score low.
CORPUS_PATH = (
    Path(__file__).resolve().parent.parent.parent / "lora-finetune/data/train.jsonl"
)

JUDGE_PROMPT = """\
Rate this Q&A pair. Return ONLY the JSON — no other text.

QUESTION: {question}

RETRIEVED CONTEXT: {context}

ANSWER: {answer}

Score each dimension 1-3:
- quality: Does the answer correctly and clearly address the question?
  3=correct and clear, 2=partially correct or vague, 1=wrong or off-topic
- grounded: Is the answer supported by the context (or sound general knowledge if no context)?
  3=well grounded, 2=one unsupported claim, 1=likely hallucination

Return: {{"quality": N, "grounded": N, "hallucination": true_or_false}}"""

_corpus_cache: list[dict] | None = None
_corpus_emb_cache: np.ndarray | None = None


def _load_corpus() -> list[dict]:
    global _corpus_cache
    if _corpus_cache is None:
        cases = []
        if CORPUS_PATH.exists():
            with open(CORPUS_PATH) as f:
                for line in f:
                    if line.strip():
                        cases.append(json.loads(line))
        _corpus_cache = cases
    return _corpus_cache


def _corpus_text(record: dict) -> str:
    """Normalize corpus record to text — handles both rag_qa and lora-finetune formats."""
    if "instruction" in record:
        return record["instruction"] + " " + record.get("output", "")
    return record.get("input", "") + " " + record.get("golden_answer", "")


def _get_corpus_embeddings(emb_module) -> np.ndarray:
    global _corpus_emb_cache
    if _corpus_emb_cache is None:
        corpus = _load_corpus()
        if corpus:
            texts = [_corpus_text(c) for c in corpus]
            _corpus_emb_cache = emb_module.embed_queries(texts)
        else:
            _corpus_emb_cache = np.array([])
    return _corpus_emb_cache


def _retrieve_context(query: str, emb_module, sim_threshold: float = 0.30) -> str:
    corpus = _load_corpus()
    corpus_embs = _get_corpus_embeddings(emb_module)
    if len(corpus) == 0 or corpus_embs.size == 0:
        return ""
    query_emb = emb_module.embed_queries([query])[0]
    norms = np.linalg.norm(corpus_embs, axis=1) * np.linalg.norm(query_emb) + 1e-9
    sims = np.dot(corpus_embs, query_emb) / norms
    top_idx = int(np.argmax(sims))
    if float(sims[top_idx]) < sim_threshold:
        return ""
    rec = corpus[top_idx]
    answer = rec.get("output") or rec.get("golden_answer", "")
    return answer[:500]


def _call_haiku(client: anthropic.Anthropic, system: str, user: str) -> str:
    resp = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=256,
        temperature=0,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return resp.content[0].text.strip()


def retrieval_quality_score(queries: list[str], emb_module) -> dict:
    """
    Measure retrieval quality for each query: cosine similarity to best match in ML corpus.

    In-distribution (ML Q&A) queries: ~0.5-0.7 similarity → retrieval succeeds.
    OOD queries: ~0.1-0.3 similarity → retrieval misses, model answers without context.

    No API calls. Deterministic. Scales to full batch (not just a sample).
    """
    corpus = _load_corpus()
    corpus_embs = _get_corpus_embeddings(emb_module)
    if len(corpus) == 0 or corpus_embs.size == 0:
        return {"avg_retrieval_sim": 0.0, "retrieval_miss_rate": 0.0}

    batch_embs = emb_module.embed_queries(queries)
    sims = []
    for qe in batch_embs:
        norms = np.linalg.norm(corpus_embs, axis=1) * np.linalg.norm(qe) + 1e-9
        cos_sims = np.dot(corpus_embs, qe) / norms
        sims.append(float(np.max(cos_sims)))

    miss_rate = sum(1 for s in sims if s < 0.30) / len(sims)
    return {
        "avg_retrieval_sim": round(float(np.mean(sims)), 4),
        "retrieval_miss_rate": round(miss_rate, 4),
    }


def score_batch_sample(
    queries: list[str],
    client: anthropic.Anthropic,
    emb_module,
    sample_size: int = 5,
) -> dict:
    """
    Sample queries from a batch, generate answers, judge quality.
    Returns avg_quality_score (1-3), hallucination_rate (0-1), n_sampled.
    """
    sampled = random.sample(queries, min(sample_size, len(queries)))
    quality_scores: list[float] = []
    halluc_flags: list[int] = []

    for query in sampled:
        context = _retrieve_context(query, emb_module)
        user_msg = (
            f"Context:\n{context}\n\nQuestion: {query}"
            if context
            else f"Question: {query}"
        )
        try:
            answer = _call_haiku(
                client,
                "You are an ML/AI assistant. Answer clearly using the context if relevant.",
                user_msg,
            )
            judge_raw = _call_haiku(
                client,
                "You are a strict evaluator. Return only valid JSON.",
                JUDGE_PROMPT.format(
                    question=query[:300],
                    context=context[:200] if context else "none retrieved",
                    answer=answer[:400],
                ),
            )
            m = re.search(r"\{.*?\}", judge_raw, re.DOTALL)
            if m:
                scores = json.loads(m.group())
                quality_scores.append(float(scores.get("quality", 2)))
                halluc_flags.append(1 if scores.get("hallucination", False) else 0)
            else:
                quality_scores.append(2.0)
                halluc_flags.append(0)
        except Exception:
            quality_scores.append(2.0)
            halluc_flags.append(0)

    return {
        "avg_quality_score": round(float(np.mean(quality_scores)), 3) if quality_scores else 0.0,
        "hallucination_rate": round(float(np.mean(halluc_flags)), 3) if halluc_flags else 0.0,
        "n_sampled": len(sampled),
    }
