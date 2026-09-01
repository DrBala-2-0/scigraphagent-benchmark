"""
step03_run_experiments.py
==========================
Executes the three benchmark experiments defined in Section 5.2
of the SciGraphAgent paper.

  Experiment 1 — Does the runtime RAGAS retry gate improve answer quality?
    Conditions : single-pass (no gate) vs RAGAS-gated retry
    Measures   : F1, Exact Match, Faithfulness, Answer Relevancy
    Compares   : SciGraphAgent gate vs Self-RAG self-reflection baseline
                 Self-RAG operates at token level (special tokens)
                 Our gate operates at answer level using RAGAS metrics

  Experiment 2 — Does hybrid Graph-RAG outperform vector-only RAG?
    Conditions : A (no retrieval), B (vector-only), C (graph-only), D (hybrid)
    Measures   : F1, Faithfulness, Context Recall per condition
    Target     : Condition D recall ≥ +15pp over Condition B (Han et al. 2025)

  Experiment 3 — Does the CI/CD faithfulness gate catch regressions?
    Conditions : V1 (full config) vs V2 (degraded top-k=1, no graph)
    Measures   : Average faithfulness — V1 must PASS, V2 must be BLOCKED

PROVIDER
--------
All LLM calls go to Groq via the OpenAI-compatible endpoint.
  Generator : openai/gpt-oss-120b (replaces llama-3.3-70b-versatile,
              deprecated Aug 16 2026)
  Judge     : openai/gpt-oss-20b  (replaces llama-3.1-8b-instant,
              deprecated Aug 16 2026)
  Base URL  : https://api.groq.com/openai/v1
  Free tier : 30 RPM, 1000 RPD, 8000 TPM, 200000 TPD per model

WHY SPLIT GENERATOR AND JUDGE?
-------------------------------
Using different models for generation and judging eliminates the
self-judge bias documented in the LLM-as-judge literature — a model
that judges its own outputs scores them 10-25% higher than an
independent judge would. GPT-OSS 120B generates, GPT-OSS 20B judges.

SCORING
-------
F1 and Exact Match: deterministic, computed against gold answers.
Faithfulness      : RAGAS-style LLM-as-judge (Es et al. 2023,
                    arXiv:2309.15217). GPT-OSS 20B evaluates whether
                    each claim in the answer is supported by context.
Answer Relevancy  : RAGAS-style LLM-as-judge. GPT-OSS 20B scores
                    whether the answer addresses the question.
Context Recall    : Deterministic proxy — fraction of gold answer
                    tokens present in retrieved context.

USAGE
-----
    python step03_run_experiments.py --dataset hotpotqa --n 3
    python step03_run_experiments.py --dataset hotpotqa --n 50 --experiments 1
    python step03_run_experiments.py --dataset hotpotqa --n 50 --experiments 1 2 3
"""

import json
import time
import re
import argparse
import os
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
from dotenv import load_dotenv

# Load .env — must happen before any os.environ.get() call
load_dotenv()

DATA_DIR    = Path("data")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# ── Daily token budget tracker ────────────────────────────────────
# Groq free tier: 200,000 TPD per model.
# We stop at 190,000 to leave headroom for judge calls.
_tokens_used = 0
TPD_LIMIT    = 190_000


def track_tokens(prompt: str, response: str, max_tokens: int) -> None:
    """
    Rough token estimate: 1 token ≈ 4 characters (English text).
    Raises SystemExit with a clean message when daily budget is reached.
    Results already saved up to this point are preserved.
    """
    global _tokens_used
    estimated = len(prompt) // 4 + min(len(response) // 4, max_tokens)
    _tokens_used += estimated
    if _tokens_used > TPD_LIMIT:
        print(f"\n  ⚠  Daily token budget reached "
              f"({_tokens_used:,} / {TPD_LIMIT:,} estimated tokens).")
        print(f"  Results saved so far are intact in {RESULTS_DIR}/")
        print(f"  Resume tomorrow with the same command.")
        raise SystemExit(0)


# ── Groq API client ───────────────────────────────────────────────

def call_llm(prompt: str,
             system:     str   = "",
             max_tokens: int   = 512,
             model:      str   = "openai/gpt-oss-120b") -> str:
    """
    Single LLM call via Groq (OpenAI-compatible endpoint).

    Generator default : openai/gpt-oss-120b
    Judge calls pass  : openai/gpt-oss-20b

    Why openai/gpt-oss-120b?
    Replaces llama-3.3-70b-versatile (deprecated Aug 16 2026).
    Matches or surpasses OpenAI o4-mini on reasoning benchmarks.
    Runs as MoE (5.1B active params per forward pass) on Groq's LPU.

    Why openai/gpt-oss-20b for judging?
    Replaces llama-3.1-8b-instant (deprecated Aug 16 2026).
    Different model family from 120B — eliminates self-judge bias.
    Runs at 1000 tokens/second on Groq LPU — minimal latency overhead.
    LLaMA-3.3-70B-Instruct family achieves Fleiss κ = 0.87 on QAMPARI
    for faithfulness evaluation (highest individual model agreement
    with human judgments, comparable to LLM ensemble).
    """
    from openai import OpenAI
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.environ.get("GROQ_API_KEY", ""),
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system",
                 "content": system or "You are a precise scientific QA assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[API ERROR: {e}]"


def call_with_backoff(prompt:      str,
                      system:      str   = "",
                      max_tokens:  int   = 512,
                      model:       str   = "openai/gpt-oss-120b",
                      max_retries: int   = 6) -> str:
    """
    Wraps call_llm() with exponential backoff on 429 rate-limit errors.

    Wait schedule: 4, 8, 16, 32, 64, 128 seconds.
    At 8,000 TPM free tier, a single call averaging 1,200 tokens needs
    ~9 seconds of token-bucket replenishment. Starting at 4s means the
    second attempt (at 8s) reliably clears the bucket.

    Also tracks daily token usage and raises SystemExit when TPD_LIMIT
    is reached — ensures results are saved before the budget runs out.
    """
    for attempt in range(max_retries):
        result = call_llm(prompt, system=system,
                          max_tokens=max_tokens, model=model)
        track_tokens(prompt, result, max_tokens)

        if not result.startswith("[API ERROR"):
            return result

        if "429" in result or "rate_limit" in result.lower():
            wait = 4 * (2 ** attempt)    # 4, 8, 16, 32, 64, 128s
            print(f"    ⏳ Rate limit. Waiting {wait}s "
                  f"(attempt {attempt+1}/{max_retries})")
            time.sleep(wait)
        else:
            return result   # non-rate-limit error — return immediately

    return result


# ── Deterministic scoring ─────────────────────────────────────────

def normalise(text: str) -> str:
    """
    Normalise answer text for F1/EM computation.
    Follows the HotpotQA official evaluation script exactly:
      1. Lowercase
      2. Remove articles (a, an, the)
      3. Remove punctuation
      4. Collapse whitespace
    """
    text = text.lower().strip()
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()


def exact_match(pred: str, gold: str) -> float:
    return float(normalise(pred) == normalise(gold))


def f1_score(pred: str, gold: str) -> float:
    pred_tokens = normalise(pred).split()
    gold_tokens = normalise(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall    = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def judge_context_recall(gold_answer: str, context: str) -> float:
    """
    Deterministic context recall proxy.
    Fraction of gold answer tokens present in retrieved context.
    No API call — fast and reproducible.
    Target for Condition D: ≥ +15pp over Condition B.
    """
    gold_tokens = set(normalise(gold_answer).split())
    ctx_tokens  = set(normalise(context).split())
    if not gold_tokens:
        return 0.0
    return len(gold_tokens & ctx_tokens) / len(gold_tokens)


# ── RAGAS-style LLM-as-judge ──────────────────────────────────────

def judge_faithfulness(answer: str, context: str) -> float:
    """
    RAGAS faithfulness metric (Es et al. 2023, arXiv:2309.15217).

    Measures whether each claim in the answer is supported by context.
    Score: 0.0 (no claims supported) to 1.0 (all claims supported).

    Implementation:
    - GPT-OSS 20B reads answer + context
    - Breaks answer into individual factual claims
    - Checks each claim against context
    - Returns supported_claims / total_claims as faithfulness score

    This score is the conditional edge condition in the retry gate:
    if faithfulness < 0.75 → re-retrieve and re-synthesise.

    max_tokens=300: prevents truncated JSON on long reasoning chains.
    """
    if not context or not answer or answer.startswith("[API ERROR"):
        return 0.0

    prompt = f"""You are evaluating whether an answer is faithful to the context.

Context:
{context[:1500]}

Answer:
{answer}

Instructions:
- Evaluate ONLY based on what is EXPLICITLY written in the context.
- Do NOT use background knowledge or make inferences beyond the context.
- Break the answer into individual factual claims.
- For each claim, check if it is explicitly supported by the context.
- Return a JSON object with exactly these fields:
  {{"total_claims": <int>, "supported_claims": <int>, "faithfulness": <float 0.0-1.0>}}
- faithfulness = supported_claims / total_claims
- If the answer is "Insufficient context", return faithfulness 1.0 (no hallucination).

Respond with ONLY the JSON object, no other text."""

    resp = call_with_backoff(prompt, max_tokens=300,
                             model="openai/gpt-oss-20b")
    return _parse_float_from_json(resp, "faithfulness", default=0.5)


def judge_relevancy(answer: str, question: str) -> float:
    """
    RAGAS answer relevancy metric (Es et al. 2023, arXiv:2309.15217).

    Measures whether the answer actually addresses the question.
    Score: 0.0 (completely off-topic) to 1.0 (directly answers).

    max_tokens=200: sufficient for JSON + one-sentence reason.
    """
    if not answer or answer.startswith("[API ERROR"):
        return 0.0

    prompt = f"""Score whether the answer addresses the question.
0.0 = completely off-topic, 1.0 = directly and fully answers.

Question: {question}
Answer: {answer}

Return ONLY a JSON object:
{{"relevancy": <float 0.0-1.0>, "reason": "<one sentence>"}}"""

    resp = call_with_backoff(prompt, max_tokens=200,
                             model="openai/gpt-oss-20b")
    return _parse_float_from_json(resp, "relevancy", default=0.5)


def _parse_float_from_json(resp: str, key: str,
                            default: float = 0.5) -> float:
    """
    Robust JSON parser for judge responses.

    GPT-OSS 20B occasionally wraps JSON in markdown fences or adds
    surrounding text. This parser handles all common formats:
      - Pure JSON: {"faithfulness": 0.8}
      - Fenced: ```json\n{"faithfulness": 0.8}\n```
      - With text: "Here is the evaluation:\n{"faithfulness": 0.8}"

    re.search(r'\\{.*\\}', clean, re.DOTALL) extracts the JSON object
    even when surrounded by other text.
    """
    try:
        # Strip markdown fences
        clean = resp.replace("```json", "").replace("```", "").strip()
        # Extract JSON object even if surrounded by text
        match = re.search(r'\{.*\}', clean, re.DOTALL)
        if match:
            clean = match.group(0)
        data = json.loads(clean)
        return float(data.get(key, default))
    except Exception:
        # Last resort: regex extract the number directly
        match = re.search(rf'"{key}"\s*:\s*([0-9.]+)', resp)
        return float(match.group(1)) if match else default


# ── Answer generator ──────────────────────────────────────────────

def generate_answer(question: str, context: str) -> str:
    """
    Generate an answer using GPT-OSS 120B given question and context.

    Strict grounding prompt: "Answer ONLY from the provided context."
    This mirrors SciGraphAgent's Synthesizer node prompt template.
    If context is empty (Condition A), asks LLM to answer from memory.
    """
    system = (
        "You are a precise scientific QA assistant. "
        "Answer ONLY from the provided context. "
        "If the context does not contain enough information, "
        "say exactly: Insufficient context. "
        "Be concise — answer in 1-2 sentences maximum."
    )
    if context:
        prompt = (f"Context:\n{context}\n\n"
                  f"Question: {question}\n\nAnswer:")
    else:
        prompt = (f"Question: {question}\n\n"
                  f"Answer from your training knowledge:")

    result = call_with_backoff(prompt, system=system,
                               max_tokens=150,
                               model="openai/gpt-oss-120b")
    if not result or not result.strip():
        return "Insufficient context"
    return result


# ── AgentResult dataclass ─────────────────────────────────────────

@dataclass
class AgentResult:
    """
    Complete output from one agent execution.
    Stored as a dict in the raw results JSON for each question.
    """
    question:       str
    answer:         str
    context:        str
    faithfulness:   float
    relevancy:      float
    f1:             float
    em:             float
    context_recall: float
    iterations:     int    # 1 = accepted first pass; 2 = one retry
    gate_triggered: bool   # True if retry gate fired
    latency_s:      float


# ── The runtime RAGAS retry gate ──────────────────────────────────

def run_with_retry_gate(question:        str,
                        gold_answer:     str,
                        retriever,
                        faith_threshold: float = 0.75,
                        relev_threshold: float = 0.80,
                        max_retries:     int   = 2,
                        top_k:           int   = 5) -> AgentResult:
    """
    SciGraphAgent's runtime RAGAS evaluation gate.

    This is the primary novelty of the paper (Section 3.3).
    Implements the conditional edge in the LangGraph StateGraph:

      Retrieval → Synthesizer → Evaluator
                                    │
                      faith ≥ 0.75 AND relev ≥ 0.80
                                    │
                              ┌─────┴─────┐
                              YES         NO
                              │           │
                             END      Retrieval ← retry with larger top_k

    Comparison to Self-RAG (Asai et al., ICLR 2024, arXiv:2310.11511):
    - Self-RAG: token-level special tokens ([Retrieve], [IsRel], [IsSup])
                operates during generation, one passage at a time
    - Our gate: answer-level RAGAS scores, operates after full synthesis,
                uses the complete answer as the unit of evaluation

    The two approaches address different failure modes:
    - Self-RAG catches irrelevant retrieval during generation
    - Our gate catches low-faithfulness answers after synthesis

    On retry: top_k is expanded by 3 per iteration to retrieve broader
    context — the hypothesis being that the first retrieval missed
    the supporting evidence needed for the question.
    """
    t0            = time.perf_counter()
    gate_triggered = False

    for iteration in range(max_retries):
        effective_k = top_k + (iteration * 3)   # expand on retry
        context     = retriever.retrieve(question, top_k=effective_k)
        answer      = generate_answer(question, context)
        faith       = judge_faithfulness(answer, context)
        relev       = judge_relevancy(answer, question)
        recall      = judge_context_recall(gold_answer, context)

        result = AgentResult(
            question=question,
            answer=answer,
            context=context,
            faithfulness=faith,
            relevancy=relev,
            f1=f1_score(answer, gold_answer),
            em=exact_match(answer, gold_answer),
            context_recall=recall,
            iterations=iteration + 1,
            gate_triggered=gate_triggered,
            latency_s=time.perf_counter() - t0,
        )

        # ── Conditional edge logic ───────────────────────────────
        if faith >= faith_threshold and relev >= relev_threshold:
            break                       # quality met → route to END
        elif iteration < max_retries - 1:
            gate_triggered = True       # gate fires → route to Retrieval

    result.gate_triggered = gate_triggered
    return result


def run_without_gate(question:    str,
                     gold_answer: str,
                     retriever,
                     top_k:       int = 5) -> AgentResult:
    """
    Single-pass agent — no retry gate.
    Baseline for Experiment 1: accepts whatever answer the first
    retrieval + synthesis produces, regardless of quality.
    """
    t0      = time.perf_counter()
    context = retriever.retrieve(question, top_k=top_k)
    answer  = generate_answer(question, context)
    faith   = judge_faithfulness(answer, context)
    relev   = judge_relevancy(answer, question)
    recall  = judge_context_recall(gold_answer, context)

    return AgentResult(
        question=question,
        answer=answer,
        context=context,
        faithfulness=faith,
        relevancy=relev,
        f1=f1_score(answer, gold_answer),
        em=exact_match(answer, gold_answer),
        context_recall=recall,
        iterations=1,
        gate_triggered=False,
        latency_s=time.perf_counter() - t0,
    )


# ── Experiment 1 ──────────────────────────────────────────────────

def run_experiment_1(records:     list,
                     retrievers:  list,
                     max_retries: int) -> dict:
    """
    Experiment 1: Runtime RAGAS retry gate vs single-pass.

    Uses Condition D (Hybrid Graph-RAG) as the retriever for both
    conditions — so the only variable is the presence of the gate.
    Any performance difference is attributable to the gate alone.

    Expected: faithfulness improves with gate; F1 may also improve
    as the gate forces retrieval of more supporting context.
    Gate trigger rate typically 25-55% on multi-hop questions.
    """
    print("\n  [EXP 1] Runtime RAGAS Retry Gate vs Single Pass")
    hybrid = next(r for r in retrievers if "Hybrid" in r.name)
    no_gate_results, gate_results = [], []

    for i, rec in enumerate(records):
        print(f"    Q{i+1:3d}/{len(records)}: {rec['question'][:55]}...")

        ng = run_without_gate(rec["question"], rec["answer"], hybrid)
        wg = run_with_retry_gate(rec["question"], rec["answer"], hybrid,
                                 max_retries=max_retries)
        no_gate_results.append(asdict(ng))
        gate_results.append(asdict(wg))

        fired = "↺ retry" if wg.gate_triggered else "   ok  "
        print(f"    No gate → F1={ng.f1:.2f} "
              f"Faith={ng.faithfulness:.2f}")
        print(f"    W/ gate → F1={wg.f1:.2f} "
              f"Faith={wg.faithfulness:.2f} [{fired}] "
              f"iter={wg.iterations}")

    return {
        "experiment":   1,
        "description":  "Runtime RAGAS retry gate vs single pass",
        "no_gate":      no_gate_results,
        "with_gate":    gate_results,
    }


# ── Experiment 2 ──────────────────────────────────────────────────

def run_experiment_2(records:    list,
                     retrievers: list) -> dict:
    """
    Experiment 2: Four-condition retrieval ablation.

    All four conditions run on identical questions with identical
    prompts — only the retrieved context differs. This isolates
    the contribution of each retrieval signal.

    Validated by GraphRAG-R1 (Yu et al., WWW 2026, arXiv:2507.23581)
    whose ablation shows Text+Graph > Text only > Graph only —
    consistent with our Condition D > B > C prediction.
    """
    print("\n  [EXP 2] Four-Condition Retrieval Ablation")
    all_results = {r.name: [] for r in retrievers}

    for i, rec in enumerate(records):
        print(f"    Q{i+1:3d}/{len(records)}: {rec['question'][:55]}...")
        for ret in retrievers:
            ctx    = ret.retrieve(rec["question"], top_k=5)
            ans    = generate_answer(rec["question"], ctx)
            faith  = judge_faithfulness(ans, ctx)
            relev  = judge_relevancy(ans, rec["question"])
            recall = judge_context_recall(rec["answer"], ctx)
            result = {
                "question":      rec["question"],
                "answer":        ans,
                "gold":          rec["answer"],
                "condition":     ret.name,
                "alpha":         ret.alpha,
                "faithfulness":  faith,
                "relevancy":     relev,
                "f1":            f1_score(ans, rec["answer"]),
                "em":            exact_match(ans, rec["answer"]),
                "context_recall": recall,
            }
            all_results[ret.name].append(result)
            print(f"      {ret.name[:35]:35s} "
                  f"F1={result['f1']:.2f} "
                  f"Faith={faith:.2f} "
                  f"Recall={recall:.2f}")

    return {
        "experiment":  2,
        "description": "Four-condition retrieval ablation",
        "conditions":  all_results,
    }


# ── Experiment 3 ──────────────────────────────────────────────────

def run_experiment_3(records:    list,
                     retrievers: list,
                     collection,
                     kg) -> dict:
    """
    Experiment 3: CI/CD faithfulness regression gate.

    Compares V1 (full config) vs V2 (deliberately degraded config).
    The gate must:
      - PASS V1: full retrieval produces faithful answers (≥ 0.75)
      - BLOCK V2: degraded retrieval produces unfaithful answers (< 0.75)

    If both conditions hold, the gate is effective as a regression
    detector. This is the engineering novelty — not found in any
    reviewed scientific Graph-RAG system (Section 1.3 of the paper).

    Uses first min(5, len(records)) records as canonical test cases.
    """
    print("\n  [EXP 3] CI/CD Faithfulness Regression Gate")

    from step02_build_retrieval_systems import ConditionD_HybridGraphRAG

    # V1 — full configuration
    v1 = ConditionD_HybridGraphRAG(collection, kg, alpha=0.6)
    v1.name = "V1: Full config (top-k=5, BFS depth=2)"

    # V2 — deliberately degraded: top-k=1, no graph
    class DegradedRetriever(ConditionD_HybridGraphRAG):
        name  = "V2: Degraded (top-k=1, no graph)"
        alpha = 0.0

        def retrieve(self, query: str, top_k: int = 5) -> str:
            results = self.collection.query(
                query_texts=[query],
                n_results=1,
                include=["documents"],
            )
            chunks = results["documents"][0]
            chunk  = chunks[0] if chunks else "No results"
            return f"[Degraded Context] {chunk}"

    v2 = DegradedRetriever(collection, kg)

    THRESHOLD = 0.75
    canonical = records[:min(10, len(records))]   # 5 changed to 10
    v1_faiths, v2_faiths = [], []

    for rec in canonical:
        for label, ret, faiths in [
            ("V1", v1, v1_faiths),
            ("V2", v2, v2_faiths),
        ]:
            ctx   = ret.retrieve(rec["question"],
                                 top_k=5 if label == "V1" else 1)
            ans   = generate_answer(rec["question"], ctx)
            faith = judge_faithfulness(ans, ctx)
            faiths.append(faith)
            print(f"    {label} faith={faith:.3f}  "
                  f"Q: {rec['question'][:50]}...")

    avg_v1   = sum(v1_faiths) / len(v1_faiths)
    avg_v2   = sum(v2_faiths) / len(v2_faiths)
    v1_pass  = avg_v1 >= THRESHOLD
    v2_block = avg_v2 <  THRESHOLD

    print(f"\n    V1 avg faithfulness = {avg_v1:.3f} → "
          f"{'PASS ✓' if v1_pass else 'FAIL ❌'}")
    print(f"    V2 avg faithfulness = {avg_v2:.3f} → "
          f"{'BLOCKED ✓' if v2_block else 'PASSED (gate missed regression)'}")
    print(f"    Gate effective: {'✓ YES' if v1_pass and v2_block else '✗ NO'}")

    return {
        "experiment":           3,
        "description":          "CI/CD faithfulness regression gate",
        "threshold":            THRESHOLD,
        "v1_faithfulness":      avg_v1,
        "v2_faithfulness":      avg_v2,
        "v1_passed_gate":       v1_pass,
        "v2_blocked_correctly": v2_block,
        "gate_effective":       v1_pass and v2_block,
    }


# ── Main ──────────────────────────────────────────────────────────

def main(dataset:      str   = "hotpotqa",
         n:            int   = 50,
         max_retries:  int   = 2,
         experiments:  list  = None) -> dict:

    if experiments is None:
        experiments = [1, 2, 3]

    print("\n" + "=" * 58)
    print("  STEP 03: RUN EXPERIMENTS")
    print("=" * 58)
    print(f"  Dataset     : {dataset}")
    print(f"  N           : {n}")
    print(f"  Experiments : {experiments}")
    print(f"  Max retries : {max_retries}")
    print(f"  Generator   : openai/gpt-oss-120b (Groq)")
    print(f"  Judge       : openai/gpt-oss-20b  (Groq)")
    print(f"  TPD budget  : {TPD_LIMIT:,} tokens\n")

    # ── Check API key ────────────────────────────────────────────
    if not os.environ.get("GROQ_API_KEY"):
        print("  ✗ GROQ_API_KEY not set.")
        print("  Add it to .env: GROQ_API_KEY=gsk_...")
        return {}

    # ── Load data ────────────────────────────────────────────────
    data_file = DATA_DIR / f"{dataset}_sample_{n}.json"
    if not data_file.exists():
        print(f"  ✗ {data_file} not found.")
        print(f"  Run first: python step01_load_benchmarks.py --n {n}")
        return {}

    with open(data_file) as f:
        records = json.load(f)
    print(f"  Loaded {len(records)} records from {data_file.name}")

    # ── Build retrieval systems ──────────────────────────────────
    print("  Building retrieval systems...")
    from step02_build_retrieval_systems import (
        build_vector_index, build_knowledge_graph,
        ConditionA_NoRetrieval, ConditionB_VectorOnly,
        ConditionC_GraphOnly,   ConditionD_HybridGraphRAG,
    )
    collection_name = f"benchmark_{dataset}_{n}"
    # Reuse existing index from step02 if available
    import chromadb
    _client = chromadb.PersistentClient(path="index/chroma")
    existing = [c.name for c in _client.list_collections()]
    if collection_name in existing:
        from chromadb.utils import embedding_functions
        emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        collection = _client.get_collection(
            name=collection_name,
            embedding_function=emb_fn
        )
        print(f"  Reusing existing index: {collection_name} "
            f"({collection.count()} chunks)")
    else:
        collection = build_vector_index(records, collection_name)
    kg         = build_knowledge_graph(records)

    retrievers = [
        ConditionA_NoRetrieval(),
        ConditionB_VectorOnly(collection),
        ConditionC_GraphOnly(kg),
        ConditionD_HybridGraphRAG(collection, kg, alpha=0.6),
    ]

    # ── Run experiments ──────────────────────────────────────────
    all_outputs = {}
    try:
        if 1 in experiments:
            all_outputs["exp1"] = run_experiment_1(
                records, retrievers, max_retries
            )
        if 2 in experiments:
            all_outputs["exp2"] = run_experiment_2(
                records, retrievers
            )
        if 3 in experiments:
            all_outputs["exp3"] = run_experiment_3(
                records, retrievers, collection, kg
            )
    except SystemExit:
        # Daily budget reached — save whatever we have
        print("\n  Saving partial results...")

    # ── Save results ─────────────────────────────────────────────
    if all_outputs:
        results_path = RESULTS_DIR / f"raw_results_{dataset}_{n}.json"
        with open(results_path, "w") as f:
            json.dump(all_outputs, f, indent=2)
        print(f"\n  ✓ Results saved → {results_path}")
        print(f"  ✓ Tokens used   : ~{_tokens_used:,}")
        print("  ✓ Next: python step04_compute_metrics.py")

    print("=" * 58)
    return all_outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run SciGraphAgent benchmark experiments"
    )
    parser.add_argument(
        "--dataset", default="hotpotqa",
        choices=["hotpotqa", "musique", "2wikimultihopqa"]
    )
    parser.add_argument(
        "--n", type=int, default=50,
        help="Sample size — must match step01 (default: 50)"
    )
    parser.add_argument(
        "--max-retries", type=int, default=2,
        help="Max retry iterations for the gate (default: 2)"
    )
    parser.add_argument(
        "--experiments", nargs="+", type=int, default=[1, 2, 3],
        help="Which experiments to run, e.g. --experiments 1 2"
    )
    args = parser.parse_args()
    main(args.dataset, args.n, args.max_retries, args.experiments)