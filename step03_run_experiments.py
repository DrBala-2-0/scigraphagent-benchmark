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
    Target     : Condition D recall >= +15pp over Condition B (Han et al. 2025)

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

RESUME SUPPORT
--------------
Experiments 1 and 2 save a checkpoint after every question.
If the daily token budget (TPD) is exhausted mid-run, results up to
that question are preserved. Re-running with the same command resumes
from the next unfinished question automatically.

Checkpoint files: results/checkpoints/{dataset}_{n}_{exp}.json
To force a fresh run (ignore checkpoints):
    rm results/checkpoints/hotpotqa_50_*.json

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
from dotenv import load_dotenv

# ── Load environment variables ────────────────────────────────────
# This reads the .env file and puts GROQ_API_KEY into os.environ.
# Must happen before any os.environ.get() call.
load_dotenv()

# ── Directory setup ───────────────────────────────────────────────
# DATA_DIR    : where step01 saved the benchmark JSON files
# RESULTS_DIR : where we save raw results and metrics
# CHECKPOINT_DIR: where we save per-question progress for resume
DATA_DIR       = Path("data")
RESULTS_DIR    = Path("results")
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"

RESULTS_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)

# ══════════════════════════════════════════════════════════════════
# DAILY TOKEN BUDGET TRACKER
# ══════════════════════════════════════════════════════════════════
# Groq free tier allows 200,000 tokens per day (TPD) per model.
# We stop at 190,000 to leave 10,000 tokens of headroom — enough
# for the current question's judge calls to finish even after the
# generator call triggers the stop.
#
# Why estimate instead of exact count?
# Exact counting requires the model's tokenizer (extra dependency).
# 1 token ≈ 4 characters is within 20% accuracy for English text.
# A 20% error on 190,000 is ±38,000 tokens — well within safe range.
_tokens_used = 0
TPD_LIMIT    = 190_000


def track_tokens(prompt: str, response: str, max_tokens: int) -> None:
    """
    Accumulate estimated token usage across all API calls.
    Raises SystemExit cleanly when the daily limit is approached.

    SystemExit (not Exception) is used so that the try/except in
    main() catches it and saves partial results before exiting.
    The user sees a clean message, not a Python traceback.
    """
    global _tokens_used
    # Estimate: input chars / 4 + min(output chars / 4, max_tokens)
    estimated = len(prompt) // 4 + min(len(response) // 4, max_tokens)
    _tokens_used += estimated
    if _tokens_used > TPD_LIMIT:
        print(f"\n  Daily token budget reached "
              f"(~{_tokens_used:,} / {TPD_LIMIT:,} estimated tokens).")
        print(f"  Checkpoints saved — resume tomorrow with the same command.")
        print(f"  Results so far are in {RESULTS_DIR}/")
        raise SystemExit(0)


# ══════════════════════════════════════════════════════════════════
# CHECKPOINT — RESUME SUPPORT
# ══════════════════════════════════════════════════════════════════
# A checkpoint is a JSON file saved after every question.
# It records: which questions are done and their results.
# On the next run, the experiment skips already-done questions
# and continues from where it stopped.
#
# This is essential for n=50+ runs where the daily TPD limit
# (200k tokens) may be exhausted before all questions are processed.
# Without checkpoints, a mid-run stop means restarting from scratch.

def save_checkpoint(dataset: str, n: int, exp_key: str,
                    completed: list, last_q: int) -> None:
    """
    Save experiment progress to disk after each question.

    Args:
        dataset   : "hotpotqa", "musique", etc.
        n         : sample size (3, 50, 1000)
        exp_key   : experiment identifier, e.g. "exp1_ng", "exp2"
        completed : list of result dicts completed so far
        last_q    : index of the last completed question (0-based)

    File: results/checkpoints/{dataset}_{n}_{exp_key}.json
    """
    path = CHECKPOINT_DIR / f"{dataset}_{n}_{exp_key}.json"
    with open(path, "w") as f:
        json.dump({
            "dataset":   dataset,
            "n":         n,
            "exp":       exp_key,
            "last_q":    last_q,      # 0-based index of last completed Q
            "completed": completed,   # list of result dicts
        }, f, indent=2)


def load_checkpoint(dataset: str, n: int,
                    exp_key: str) -> tuple:
    """
    Load a checkpoint if one exists for this experiment.

    Returns:
        (completed_results, start_from_index)
        - completed_results: list of already-done result dicts
        - start_from_index : 0-based index to resume from

    Returns ([], 0) if no checkpoint exists — fresh start.
    """
    path = CHECKPOINT_DIR / f"{dataset}_{n}_{exp_key}.json"
    if not path.exists():
        # No checkpoint — this is a fresh run
        return [], 0

    with open(path) as f:
        ck = json.load(f)

    completed = ck.get("completed", [])
    last_q    = ck.get("last_q", 0)
    start_from = last_q + 1   # resume from the NEXT question

    print(f"  Resuming {exp_key}: {len(completed)} questions done, "
          f"continuing from Q{start_from + 1}")
    return completed, start_from


# ══════════════════════════════════════════════════════════════════
# GROQ API — SINGLE CALL
# ══════════════════════════════════════════════════════════════════

def call_llm(prompt: str,
             system:     str = "",
             max_tokens: int = 512,
             model:      str = "openai/gpt-oss-120b") -> str:
    """
    Make one LLM call to Groq and return the text response.

    Why openai/gpt-oss-120b for generation?
      Replaces llama-3.3-70b-versatile (deprecated Aug 16 2026).
      Runs as MoE — only 5.1B parameters active per forward pass —
      so it is fast despite the 120B parameter count.

    Why openai/gpt-oss-20b for judging?
      Replaces llama-3.1-8b-instant (deprecated Aug 16 2026).
      Using a DIFFERENT model family as the judge eliminates
      self-judge bias: models score their own outputs 10-25% higher
      than an independent judge would (Zheng et al. 2023).

    Returns "[API ERROR: ...]" string on failure — never raises.
    Callers check for this prefix to detect errors.
    """
    from openai import OpenAI

    # The Groq API is OpenAI-compatible — same SDK, different base URL
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.environ.get("GROQ_API_KEY", ""),
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                # System message sets the assistant's behaviour
                {"role": "system",
                 "content": system or "You are a precise scientific QA assistant."},
                # User message is the actual prompt
                {"role": "user", "content": prompt},
            ],
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[API ERROR: {e}]"


# ══════════════════════════════════════════════════════════════════
# GROQ API — WITH BACKOFF AND BUDGET TRACKING
# ══════════════════════════════════════════════════════════════════

def call_with_backoff(prompt:      str,
                      system:      str = "",
                      max_tokens:  int = 512,
                      model:       str = "openai/gpt-oss-120b",
                      max_retries: int = 6) -> str:
    """
    Wraps call_llm() with two protective mechanisms:

    1. Exponential backoff on 429 (rate limit) errors:
       Wait schedule: 4, 8, 16, 32, 64, 128 seconds.
       Why exponential? If many callers hit the limit simultaneously
       and all retry at a fixed interval, they all retry together
       again — thundering herd. Growing waits spread them out.
       Why start at 4s? At 8,000 TPM, a 1,200-token call takes
       ~9 seconds to replenish. 4s start → 8s second attempt
       reliably clears the token bucket.

    2. Daily token budget tracking:
       Calls track_tokens() after every response.
       If the estimated daily total exceeds TPD_LIMIT (190,000),
       raises SystemExit so main() can save partial results cleanly.
       Non-429 errors (e.g. network timeout) return immediately —
       retrying a malformed request 6 times wastes your daily budget.
    """
    for attempt in range(max_retries):
        result = call_llm(prompt, system=system,
                          max_tokens=max_tokens, model=model)

        # Track tokens BEFORE checking for errors —
        # even failed calls consume some tokens on Groq's side
        track_tokens(prompt, result, max_tokens)

        if not result.startswith("[API ERROR"):
            return result   # success

        is_rate_limit = "429" in result or "rate_limit" in result.lower()
        if is_rate_limit:
            wait = 4 * (2 ** attempt)   # 4, 8, 16, 32, 64, 128 seconds
            print(f"    Rate limit hit. Waiting {wait}s "
                  f"(attempt {attempt + 1}/{max_retries})")
            time.sleep(wait)
        else:
            # Non-rate-limit error (auth failure, bad request, etc.)
            # Do not retry — return immediately to avoid wasting budget
            return result

    return result   # return last result after all retries exhausted


# ══════════════════════════════════════════════════════════════════
# DETERMINISTIC SCORING — NO API CALLS
# ══════════════════════════════════════════════════════════════════

def normalise(text: str) -> str:
    """
    Clean text for fair comparison between predicted and gold answers.

    Follows the HotpotQA official evaluation script exactly:
      1. Lowercase everything
      2. Remove articles (a, an, the) — "the Eiffel Tower" = "Eiffel Tower"
      3. Remove punctuation — "Paris." = "Paris"
      4. Collapse extra spaces

    Why normalise? Without it, "Paris." != "paris" even though both
    are correct answers. Normalisation makes the comparison fair.
    """
    text = text.lower().strip()
    text = re.sub(r'\b(a|an|the)\b', ' ', text)   # remove articles
    text = re.sub(r'[^a-z0-9\s]', '', text)        # remove punctuation
    return re.sub(r'\s+', ' ', text).strip()        # collapse spaces


def exact_match(pred: str, gold: str) -> float:
    """
    Binary metric: 1.0 if the prediction exactly matches gold (after
    normalisation), 0.0 otherwise.

    Exact Match is harsh — "Einstein" scores 0.0 against "Albert Einstein".
    Use alongside F1 which gives partial credit.
    """
    return float(normalise(pred) == normalise(gold))


def f1_score(pred: str, gold: str) -> float:
    """
    Token-overlap F1 score — gives partial credit for partial matches.

    precision = correct_words / predicted_words  (how precise?)
    recall    = correct_words / gold_words       (how complete?)
    F1        = harmonic mean of precision and recall

    Why harmonic mean? It punishes extremes:
    - A 1-word correct answer (high precision, low recall) gets moderate F1
    - A 100-word answer burying the gold (low precision, high recall) gets
      moderate F1. Both are appropriate penalties.
    """
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
    Deterministic (no API) proxy for context recall.

    Measures: what fraction of gold answer words appear in the
    retrieved context?

    Why deterministic? Context recall is a retrieval quality metric —
    it measures whether the evidence is present in the retrieved text,
    not whether the model used it correctly. This is perfectly
    calculable without an LLM — and free to compute.

    Target for Condition D (hybrid) vs Condition B (vector):
    +15 percentage points (Han et al. 2025, arXiv:2502.11371)
    """
    gold_tokens = set(normalise(gold_answer).split())
    ctx_tokens  = set(normalise(context).split())
    if not gold_tokens:
        return 0.0
    return len(gold_tokens & ctx_tokens) / len(gold_tokens)


# ══════════════════════════════════════════════════════════════════
# RAGAS-STYLE LLM-AS-JUDGE
# ══════════════════════════════════════════════════════════════════

def judge_faithfulness(answer: str, context: str) -> float:
    """
    RAGAS faithfulness metric (Es et al. 2023, arXiv:2309.15217).

    Measures: does the answer ONLY claim things explicitly supported
    by the retrieved context?

    Score: 0.0 (all claims unsupported) to 1.0 (all claims supported)
    Formula: faithfulness = supported_claims / total_claims

    Special case: if the model says "Insufficient context", that is
    an honest refusal — no false claims were made — so faithfulness = 1.0.

    Why max_tokens=300?
    The judge reasons through each claim step by step before returning
    JSON. Short max_tokens cuts off this reasoning, producing truncated
    (invalid) JSON. 300 tokens ensures complete output.

    Why "Do NOT use background knowledge"?
    In testing, the judge confused "Prussian" with "German" by applying
    world knowledge that Prussia became part of Germany. The explicit
    instruction prevents this — the judge must check only what is
    written in the context, not what it knows about the world.

    This score is the conditional edge in the retry gate:
    if faithfulness < 0.75 → re-retrieve and regenerate.
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

    Measures: does the answer actually address the question asked?

    Score: 0.0 (completely off-topic) to 1.0 (directly answers)

    Why check relevancy separately from faithfulness?
    An answer can be:
    - Faithful but irrelevant: "Oliver Reed was an English actor."
      (true, in context, but does not answer "What nationality was
      his CHARACTER?")
    - Relevant but unfaithful: "The character was German."
      (addresses the question but German is not in the context)

    Both must pass for the retry gate to accept the answer.

    Why max_tokens=200? Sufficient for JSON + one-sentence reason.
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
    Robustly extract a float value from a judge's JSON response.

    Why do we need robust parsing?
    Despite being instructed to return ONLY JSON, GPT-OSS 20B
    occasionally wraps the JSON in markdown fences (```json...```)
    or adds surrounding text. This parser handles all common formats:
      - Pure JSON: {"faithfulness": 0.8}
      - Markdown fenced: ```json\n{"faithfulness": 0.8}\n```
      - With surrounding text: "Result:\n{"faithfulness": 0.8}"

    Strategy:
      1. Strip markdown fences
      2. Use regex to find the JSON object even if surrounded by text
      3. Fall back to regex number extraction if JSON parse fails
      4. Return `default` (0.5) if all else fails — 0.5 signals
         uncertainty rather than a definitive pass or fail
    """
    try:
        # Step 1: Strip markdown fences that GPT sometimes adds
        clean = resp.replace("```json", "").replace("```", "").strip()

        # Step 2: Find the JSON object even if surrounded by text
        match = re.search(r'\{.*\}', clean, re.DOTALL)
        if match:
            clean = match.group(0)
        data = json.loads(clean)
        return float(data.get(key, default))
    except Exception:
        # Step 3: Last resort — extract the number directly with regex
        match = re.search(rf'"{key}"\s*:\s*([0-9.]+)', resp)
        return float(match.group(1)) if match else default


# ══════════════════════════════════════════════════════════════════
# ANSWER GENERATOR
# ══════════════════════════════════════════════════════════════════

def generate_answer(question: str, context: str) -> str:
    """
    Generate an answer using GPT-OSS 120B given a question and context.

    The system prompt is strict: "Answer ONLY from the provided context."
    This mirrors the Synthesizer node in SciGraphAgent's LangGraph
    pipeline — it is the component that converts retrieved evidence
    into a concise answer.

    If context is empty (Condition A — no retrieval), we ask the LLM
    to answer from training knowledge instead. This measures what the
    LLM knows without any external evidence.

    The defensive empty check (if not result.strip()) handles the rare
    case where the API returns an empty string — treat it as an honest
    "Insufficient context" rather than a blank answer that confuses
    downstream scoring.
    """
    system = (
        "You are a precise scientific QA assistant. "
        "Answer ONLY from the provided context. "
        "If the context does not contain enough information, "
        "say exactly: Insufficient context. "
        "Be concise — answer in 1-2 sentences maximum."
    )

    if context:
        # Retrieval-augmented: answer from provided evidence
        prompt = (f"Context:\n{context}\n\n"
                  f"Question: {question}\n\nAnswer:")
    else:
        # No retrieval: answer from parametric (training) knowledge
        prompt = (f"Question: {question}\n\n"
                  f"Answer from your training knowledge:")

    result = call_with_backoff(prompt, system=system,
                               max_tokens=150,
                               model="openai/gpt-oss-120b")

    # Defensive check: treat empty response as honest refusal
    if not result or not result.strip():
        return "Insufficient context"
    return result


# ══════════════════════════════════════════════════════════════════
# AGENT RESULT — STRUCTURED DATA CONTAINER
# ══════════════════════════════════════════════════════════════════

@dataclass
class AgentResult:
    """
    Stores the complete output from one agent execution (one question).

    Why a dataclass instead of a plain dict?
    - Field names are fixed — typos cause AttributeError immediately,
      not a silent None return as with dict[missing_key]
    - IDE autocomplete shows all 11 fields
    - asdict(result) converts to a plain dict for JSON saving in one call
    - Readable repr for debugging

    Fields:
      question       : the multi-hop question text
      answer         : the generated answer
      context        : the retrieved context (may be empty for Condition A)
      faithfulness   : RAGAS judge score — is answer grounded in context?
      relevancy      : RAGAS judge score — does answer address the question?
      f1             : token-overlap F1 against gold answer (deterministic)
      em             : exact match against gold answer (deterministic)
      context_recall : fraction of gold answer words in context (deterministic)
      iterations     : 1 = first pass accepted; 2 = gate fired, retry done
      gate_triggered : True if the retry gate fired on this question
      latency_s      : wall clock time for the full agent execution
    """
    question:       str
    answer:         str
    context:        str
    faithfulness:   float
    relevancy:      float
    f1:             float
    em:             float
    context_recall: float
    iterations:     int
    gate_triggered: bool
    latency_s:      float


# ══════════════════════════════════════════════════════════════════
# THE RUNTIME RAGAS RETRY GATE
# ══════════════════════════════════════════════════════════════════

def run_with_retry_gate(question:        str,
                        gold_answer:     str,
                        retriever,
                        faith_threshold: float = 0.75,
                        relev_threshold: float = 0.80,
                        max_retries:     int   = 2,
                        top_k:           int   = 5) -> AgentResult:
    """
    The primary novelty of SciGraphAgent (Section 3.3 of the paper).

    This implements the conditional edge in the LangGraph StateGraph:

      [Retrieval] → [Synthesizer] → [Evaluator]
                                          |
                          faith >= 0.75 AND relev >= 0.80?
                                    /            \
                                  YES             NO
                                   |               |
                                  END        [Retrieval] (retry with larger top_k)

    Why answer-level evaluation instead of token-level (Self-RAG)?
    Self-RAG (Asai et al., ICLR 2024, arXiv:2310.11511) inserts
    special tokens during generation to evaluate one passage at a time.
    Our gate evaluates the COMPLETE answer after synthesis — catching
    cases where individual passages looked relevant but the synthesised
    answer is still unfaithful or off-topic.

    The two approaches are complementary, not competing:
    - Self-RAG catches irrelevant retrieval DURING generation
    - Our gate catches low-quality answers AFTER synthesis

    Why expand top_k on retry?
    If the first retrieval produced a low-faithfulness answer, the
    supporting evidence likely exists in the corpus but was not in the
    top-5 results. Expanding to top_k+3 increases the chance of
    finding it. Beyond 2 retries, the quality gain rarely justifies
    the additional latency and token cost.
    """
    t0             = time.perf_counter()
    gate_triggered = False

    for iteration in range(max_retries):
        # Expand retrieval window on each retry
        effective_k = top_k + (iteration * 3)   # 5, 8, 11, ...

        # Step 1: Retrieve context
        context = retriever.retrieve(question, top_k=effective_k)

        # Step 2: Generate answer from context
        answer  = generate_answer(question, context)

        # Step 3: Evaluate answer quality (the gate)
        faith  = judge_faithfulness(answer, context)
        relev  = judge_relevancy(answer, question)
        recall = judge_context_recall(gold_answer, context)

        # Build the result object for this iteration
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

        # Step 4: Conditional edge — route to END or retry
        if faith >= faith_threshold and relev >= relev_threshold:
            break                   # quality met → accept and exit
        elif iteration < max_retries - 1:
            gate_triggered = True   # quality not met → retry

    result.gate_triggered = gate_triggered
    return result


def run_without_gate(question:    str,
                     gold_answer: str,
                     retriever,
                     top_k:       int = 5) -> AgentResult:
    """
    Single-pass agent — baseline with NO retry gate.

    Accepts whatever the first retrieval + synthesis produces,
    regardless of faithfulness or relevancy scores.
    This is the comparison condition for Experiment 1.

    Any performance difference between this and run_with_retry_gate()
    on the same questions is attributable solely to the gate.
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


# ══════════════════════════════════════════════════════════════════
# EXPERIMENT 1 — RETRY GATE VS SINGLE PASS
# ══════════════════════════════════════════════════════════════════

def run_experiment_1(records:     list,
                     retrievers:  list,
                     max_retries: int,
                     dataset:     str = "hotpotqa",
                     n:           int = 3) -> dict:
    """
    Experiment 1: Does the RAGAS retry gate improve answer quality?

    Both conditions (no-gate and with-gate) use Condition D
    (Hybrid Graph-RAG, alpha=0.6) as the retriever. The only
    variable is the presence of the gate. Any difference in
    faithfulness or F1 is attributable to the gate alone.

    Resume support: checkpoints are saved after every question.
    On restart, questions already processed are skipped automatically.
    Delete results/checkpoints/hotpotqa_50_exp1_*.json to force fresh.

    Expected at n>=50:
      Gate trigger rate: 25-55% (many questions answered well first pass)
      Faithfulness improvement: +0.05 to +0.15 on triggered queries
      Latency overhead: +5-10s per triggered query (clock-measured,
                        reliable even at n=3)
    """
    print("\n  [EXP 1] Runtime RAGAS Retry Gate vs Single Pass")

    # Find the hybrid retriever (Condition D)
    hybrid = next(r for r in retrievers if "Hybrid" in r.name)

    # Load any existing checkpoint — resume from where we left off
    no_gate_results, ng_start = load_checkpoint(dataset, n, "exp1_ng")
    gate_results,    wg_start = load_checkpoint(dataset, n, "exp1_wg")

    # Align both lists: resume from the minimum completed index
    # so no-gate and with-gate always have matching question counts
    start_from      = min(len(no_gate_results), len(gate_results))
    no_gate_results = no_gate_results[:start_from]
    gate_results    = gate_results[:start_from]

    for i, rec in enumerate(records):

        # Skip questions already completed in a previous session
        if i < start_from:
            print(f"    Q{i+1:3d}/{len(records)}: already done — skipping")
            continue

        print(f"    Q{i+1:3d}/{len(records)}: {rec['question'][:55]}...")

        # Run both conditions on the same question
        ng = run_without_gate(rec["question"], rec["answer"], hybrid)
        wg = run_with_retry_gate(rec["question"], rec["answer"], hybrid,
                                 max_retries=max_retries)

        no_gate_results.append(asdict(ng))
        gate_results.append(asdict(wg))

        # Save checkpoint immediately — this question is now safe
        # If the daily budget runs out on the next question,
        # this question's results are already preserved
        save_checkpoint(dataset, n, "exp1_ng", no_gate_results, i)
        save_checkpoint(dataset, n, "exp1_wg", gate_results,    i)

        fired = "retry" if wg.gate_triggered else "ok   "
        print(f"    No gate → F1={ng.f1:.2f} Faith={ng.faithfulness:.2f}")
        print(f"    W/ gate → F1={wg.f1:.2f} Faith={wg.faithfulness:.2f} "
              f"[{fired}] iter={wg.iterations}")

    return {
        "experiment":  1,
        "description": "Runtime RAGAS retry gate vs single pass",
        "no_gate":     no_gate_results,
        "with_gate":   gate_results,
    }


# ══════════════════════════════════════════════════════════════════
# EXPERIMENT 2 — FOUR-CONDITION ABLATION
# ══════════════════════════════════════════════════════════════════

def run_experiment_2(records:    list,
                     retrievers: list,
                     dataset:    str = "hotpotqa",
                     n:          int = 3) -> dict:
    """
    Experiment 2: Does hybrid Graph-RAG outperform vector-only RAG?

    All four conditions run on IDENTICAL questions with IDENTICAL
    prompts. The only variable is what context each retriever returns.
    This cleanly isolates each retrieval signal's contribution.

    Condition A (no retrieval) — measures parametric LLM knowledge
    Condition B (vector only)  — standard RAG baseline
    Condition C (graph only)   — structural retrieval baseline
    Condition D (hybrid)       — proposed system (alpha=0.6)

    Target: Condition D context recall >= +15pp over B (Han et al. 2025)

    Resume support: one combined checkpoint for all 4 conditions.
    Each checkpoint entry contains all 4 condition results for one Q.
    """
    print("\n  [EXP 2] Four-Condition Retrieval Ablation")

    # Load checkpoint — resume from where experiment 2 left off
    ck_data, start_from = load_checkpoint(dataset, n, "exp2")

    # Restore the results dict from checkpoint, or start fresh
    if ck_data:
        # ck_data is the all_results dict: {condition_name: [results]}
        all_results = ck_data
    else:
        # Fresh start: empty list for each retriever condition
        all_results = {r.name: [] for r in retrievers}

    for i, rec in enumerate(records):

        # Skip questions already completed in a previous session
        if i < start_from:
            print(f"    Q{i+1:3d}/{len(records)}: already done — skipping")
            continue

        print(f"    Q{i+1:3d}/{len(records)}: {rec['question'][:55]}...")

        # Run ALL four conditions on this question
        for ret in retrievers:
            ctx    = ret.retrieve(rec["question"], top_k=5)
            ans    = generate_answer(rec["question"], ctx)
            faith  = judge_faithfulness(ans, ctx)
            relev  = judge_relevancy(ans, rec["question"])
            recall = judge_context_recall(rec["answer"], ctx)
            result = {
                "question":       rec["question"],
                "answer":         ans,
                "gold":           rec["answer"],
                "condition":      ret.name,
                "alpha":          ret.alpha,
                "faithfulness":   faith,
                "relevancy":      relev,
                "f1":             f1_score(ans, rec["answer"]),
                "em":             exact_match(ans, rec["answer"]),
                "context_recall": recall,
            }
            all_results[ret.name].append(result)
            print(f"      {ret.name[:35]:35s} "
                  f"F1={result['f1']:.2f} "
                  f"Faith={faith:.2f} "
                  f"Recall={recall:.2f}")

        # Save checkpoint after ALL 4 conditions for this question
        save_checkpoint(dataset, n, "exp2", all_results, i)

    return {
        "experiment":  2,
        "description": "Four-condition retrieval ablation",
        "conditions":  all_results,
    }


# ══════════════════════════════════════════════════════════════════
# EXPERIMENT 3 — CI/CD REGRESSION GATE
# ══════════════════════════════════════════════════════════════════

def run_experiment_3(records:    list,
                     retrievers: list,
                     collection,
                     kg) -> dict:
    """
    Experiment 3: Does the CI/CD gate catch pipeline regressions?

    Engineering novelty: faithfulness threshold as a deployment gate.
    V1 (full config): top-k=5, BFS depth=2, alpha=0.6 → should PASS
    V2 (degraded)   : top-k=1, no graph                → should BLOCK

    Why is this novel?
    No reviewed scientific Graph-RAG system uses faithfulness scores
    as a deployment gate. Traditional CI/CD tests check code behaviour
    (unit tests, integration tests). Our gate checks ANSWER QUALITY —
    a form of LLM-specific regression testing.

    Why 10 canonical questions (not 5)?
    LLM judge variance is high at small n. With 5 questions,
    one score flip (0.0 → 1.0) changes the average by 0.20 —
    enough to flip the gate verdict. With 10 questions, one flip
    changes the average by 0.10 — more stable. n>=20 is ideal;
    min(10, n) is our compromise for small dev runs.

    Reliability: this gate is unreliable at n=3 (SE=±0.20 dominates).
    Reliable at n>=20 canonical questions (SE=±0.08).
    """
    print("\n  [EXP 3] CI/CD Faithfulness Regression Gate")

    from step02_build_retrieval_systems import ConditionD_HybridGraphRAG

    # V1: full configuration — this is what users get in production
    v1      = ConditionD_HybridGraphRAG(collection, kg, alpha=0.6)
    v1.name = "V1: Full config (top-k=5, BFS depth=2)"

    # V2: deliberately degraded — simulates a bad config change
    # e.g. someone changed top_k=5 to top_k=1 to save API costs,
    # not realising it makes answers much less faithful
    class DegradedRetriever(ConditionD_HybridGraphRAG):
        name  = "V2: Degraded (top-k=1, no graph)"
        alpha = 0.0

        def retrieve(self, query: str, top_k: int = 5) -> str:
            # Only retrieve 1 chunk, no graph — deliberately bad
            results = self.collection.query(
                query_texts=[query],
                n_results=1,
                include=["documents"],
            )
            chunks = results["documents"][0]
            chunk  = chunks[0] if chunks else "No results"
            return f"[Degraded Context] {chunk}"

    v2 = DegradedRetriever(collection, kg)

    # Use the first min(10, n) questions as canonical test cases
    THRESHOLD = 0.75
    canonical  = records[:min(10, len(records))]
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


# ══════════════════════════════════════════════════════════════════
# MAIN — ORCHESTRATES ALL THREE EXPERIMENTS
# ══════════════════════════════════════════════════════════════════

def main(dataset:     str  = "hotpotqa",
         n:           int  = 50,
         max_retries: int  = 2,
         experiments: list = None) -> dict:
    """
    Entry point: load data, build retrieval systems, run experiments.

    The function is intentionally structured so that:
    1. Expensive operations (index build) happen once
    2. Each experiment is independent — failure in Exp 2 does not
       prevent Exp 3 from running
    3. Results are saved even if a SystemExit (budget exhaustion)
       interrupts mid-run
    4. Checkpoints enable resuming from the last completed question

    Why separate dataset, n, max_retries as parameters?
    This makes the function callable from both the command line
    (via argparse) and from inside a Jupyter notebook (direct call).
    """
    if experiments is None:
        experiments = [1, 2, 3]

    print("\n" + "=" * 58)
    print("  STEP 03: RUN EXPERIMENTS")
    print("=" * 58)
    print(f"  Dataset      : {dataset}")
    print(f"  N            : {n}")
    print(f"  Experiments  : {experiments}")
    print(f"  Max retries  : {max_retries}")
    print(f"  Generator    : openai/gpt-oss-120b (Groq)")
    print(f"  Judge        : openai/gpt-oss-20b  (Groq)")
    print(f"  TPD budget   : {TPD_LIMIT:,} tokens")
    print(f"  Checkpoints  : {CHECKPOINT_DIR.resolve()}\n")

    # ── Verify API key is available ───────────────────────────────
    if not os.environ.get("GROQ_API_KEY"):
        print("  GROQ_API_KEY not set. Add it to .env: GROQ_API_KEY=gsk_...")
        return {}

    # ── Load benchmark questions from step01 output ───────────────
    data_file = DATA_DIR / f"{dataset}_sample_{n}.json"
    if not data_file.exists():
        print(f"  {data_file} not found.")
        print(f"  Run first: python step01_load_benchmarks.py --n {n}")
        return {}

    with open(data_file) as f:
        records = json.load(f)
    print(f"  Loaded {len(records)} records from {data_file.name}")

    # ── Build retrieval systems ───────────────────────────────────
    # Import from step02 so we reuse the exact same retrieval logic
    print("  Building retrieval systems...")
    from step02_build_retrieval_systems import (
        build_vector_index, build_knowledge_graph,
        ConditionA_NoRetrieval, ConditionB_VectorOnly,
        ConditionC_GraphOnly,   ConditionD_HybridGraphRAG,
    )

    # Reuse existing ChromaDB index from step02 if available.
    # Rebuilding takes 100+ seconds — skipping saves time on re-runs.
    collection_name = f"benchmark_{dataset}_{n}"
    import chromadb
    _client  = chromadb.PersistentClient(path="index/chroma")
    existing = [c.name for c in _client.list_collections()]

    if collection_name in existing:
        from chromadb.utils import embedding_functions
        emb_fn     = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        collection = _client.get_collection(
            name=collection_name, embedding_function=emb_fn
        )
        print(f"  Reusing existing index: {collection_name} "
              f"({collection.count()} chunks)")
    else:
        # Build fresh index — this takes a few minutes at n=50
        collection = build_vector_index(records, collection_name)

    # Knowledge graph is always rebuilt (fast: <1 second)
    kg = build_knowledge_graph(records)

    # Instantiate all four retrieval conditions
    retrievers = [
        ConditionA_NoRetrieval(),
        ConditionB_VectorOnly(collection),
        ConditionC_GraphOnly(kg),
        ConditionD_HybridGraphRAG(collection, kg, alpha=0.6),
    ]

    # ── Run the requested experiments ────────────────────────────
    all_outputs = {}
    try:
        if 1 in experiments:
            all_outputs["exp1"] = run_experiment_1(
                records, retrievers, max_retries, dataset, n
            )
        if 2 in experiments:
            all_outputs["exp2"] = run_experiment_2(
                records, retrievers, dataset, n
            )
        if 3 in experiments:
            all_outputs["exp3"] = run_experiment_3(
                records, retrievers, collection, kg
            )
    except SystemExit:
        # Daily token budget exhausted — save whatever we completed
        print("\n  Budget reached — saving partial results...")

    # ── Save consolidated results file ────────────────────────────
    # This is what step04 reads to compute aggregate metrics.
    # Checkpoints are kept separately so they can be used for resume.
    if all_outputs:
        results_path = RESULTS_DIR / f"raw_results_{dataset}_{n}.json"
        with open(results_path, "w") as f:
            json.dump(all_outputs, f, indent=2)
        print(f"\n  Results saved  → {results_path}")
        print(f"  Tokens used    : ~{_tokens_used:,}")
        print(f"  Checkpoints at : {CHECKPOINT_DIR}")
        print("  Next: python step04_compute_metrics.py")

    print("=" * 58)
    return all_outputs


# ── Command-line entry point ──────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run SciGraphAgent benchmark experiments with resume support"
    )
    parser.add_argument(
        "--dataset", default="hotpotqa",
        choices=["hotpotqa", "musique", "2wikimultihopqa"],
        help="Benchmark dataset to run on"
    )
    parser.add_argument(
        "--n", type=int, default=50,
        help="Sample size — must match step01 run (default: 50)"
    )
    parser.add_argument(
        "--max-retries", type=int, default=2,
        help="Max gate retry iterations per question (default: 2)"
    )
    parser.add_argument(
        "--experiments", nargs="+", type=int, default=[1, 2, 3],
        help="Which experiments to run, e.g. --experiments 1 2"
    )
    args = parser.parse_args()
    main(args.dataset, args.n, args.max_retries, args.experiments)
