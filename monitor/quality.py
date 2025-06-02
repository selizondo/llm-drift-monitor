"""
quality.py — Per-batch output quality monitoring without ground-truth labels.

Approach:
  1. Sample N queries from the batch.
  2. Retrieve the most relevant case from the ML Q&A corpus (Project 04 eval cases)
     using cosine similarity. Out-of-domain queries fail retrieval (sim < 0.3) → no context.
  3. Generate an answer via claude-haiku-4-5 (with or without context).
  4. Self-judge the answer on quality (1-3) and hallucination risk.

The quality degradation is structural: OOD queries fail retrieval → model answers
without grounding → judge scores drop. No simulation needed.
"""

import json
import os
import random
import re
from pathlib import Path

import anthropic
import numpy as np

CORPUS_PATH = (
    Path(__file__).resolve().parent.parent.parent / "llm-eval-harness/evals/cases/rag_qa.jsonl"
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


def _get_corpus_embeddings(emb_module) -> np.ndarray:
    global _corpus_emb_cache
    if _corpus_emb_cache is None:
        corpus = _load_corpus()
        if corpus:
            texts = [c["input"] + " " + c["golden_answer"] for c in corpus]
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
    return corpus[top_idx]["golden_answer"][:500]


def _call_haiku(client: anthropic.Anthropic, system: str, user: str) -> str:
    resp = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=256,
        temperature=0,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return resp.content[0].text.strip()


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
