"""
step06_dashboard.py
====================
Interactive Streamlit dashboard for SciGraphAgent benchmark results.

NO API CALLS — reads results/metrics_{dataset}_{n}.json from step04.

USAGE
-----
    streamlit run step06_dashboard.py
    streamlit run step06_dashboard.py -- --dataset hotpotqa --n 3

SECTIONS
--------
  0. System overview — what is being tested, four condition cards
  1. Experiment 1 — retry gate metric cards, Plotly bar chart
  2. Experiment 2 — ablation table, interactive grouped bar chart
  3. Experiment 3 — dual gauge chart, pass/fail verdict
  4. Literature comparison table with arXiv links
  5. Per-question drill-down table

DESIGN DECISIONS
----------------
- Streamlit: pip-installable, no server setup, runs locally
- Plotly: interactive charts (hover, zoom) vs static matplotlib
- Sidebar: dataset/n selectors, threshold sliders
- Cached data loading: @st.cache_data prevents re-reading on slider change
- All thresholds adjustable: student can see how gate verdict changes
"""

import json
import sys
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# ── Page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="SciGraphAgent Benchmark",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

RESULTS_DIR = Path("results")

# ── Colour palette (Wong 2011 — colour-blind safe) ─────────────────
PALETTE = {
    "A": "#000000",   # black  — no retrieval
    "B": "#56B4E9",   # sky    — vector only
    "C": "#E69F00",   # orange — graph only
    "D": "#0072B2",   # blue   — hybrid (proposed)
    "gate":    "#009E73",   # green
    "no_gate": "#56B4E9",   # sky
    "v1":      "#009E73",   # green — full config
    "v2":      "#D55E00",   # red   — degraded
    "thresh":  "#CC79A7",   # pink  — threshold line
}


# ── Data loading ───────────────────────────────────────────────────

@st.cache_data
def load_metrics(dataset: str, n: int) -> dict:
    path = RESULTS_DIR / f"metrics_{dataset}_{n}.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_raw(dataset: str, n: int) -> dict:
    path = RESULTS_DIR / f"raw_results_{dataset}_{n}.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def mean_field(results: list, field: str) -> float:
    vals = [r.get(field, 0.0) for r in results]
    return sum(vals) / len(vals) if vals else 0.0


# ── Sidebar ────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🔬 SciGraphAgent")
    st.markdown("**Benchmark Dashboard**")
    st.divider()

    dataset = st.selectbox(
        "Dataset",
        ["hotpotqa", "musique", "2wikimultihopqa"],
        index=0,
    )
    n = st.selectbox(
        "Sample size (n)",
        [3, 50, 1000],
        index=0,
    )

    st.divider()
    st.markdown("**Gate Thresholds**")
    faith_thresh = st.slider(
        "Faithfulness threshold",
        min_value=0.50, max_value=0.95,
        value=0.75, step=0.05,
        help="Retry gate fires if faithfulness < this value"
    )
    relev_thresh = st.slider(
        "Relevancy threshold",
        min_value=0.50, max_value=0.95,
        value=0.80, step=0.05,
        help="Retry gate fires if relevancy < this value"
    )
    recall_target = st.slider(
        "Recall lift target",
        min_value=0.05, max_value=0.30,
        value=0.15, step=0.05,
        help="Target recall lift: Condition D over Condition B"
    )

    st.divider()
    st.markdown("**About**")
    st.markdown(
        "SciGraphAgent combines hybrid Graph-RAG with a runtime "
        "RAGAS evaluation gate. GPU-free, open-source, pip-installable."
    )
    st.markdown("[GitHub](https://github.com/DrBala-2-0/scigraphagent-benchmark)")

# ── Load data ──────────────────────────────────────────────────────
metrics = load_metrics(dataset, n)
raw     = load_raw(dataset, n)

if not metrics:
    st.error(
        f"No metrics found for {dataset} n={n}. "
        f"Run: `python step04_compute_metrics.py --dataset {dataset} --n {n}`"
    )
    st.stop()

# ── Header ─────────────────────────────────────────────────────────
st.title("🔬 SciGraphAgent Benchmark Results")
st.markdown(
    f"**Dataset:** `{dataset.upper()}` | "
    f"**Sample size:** `n={n}` | "
    f"**Note:** n=3 is a smoke test — results reliable at n≥50"
)
st.divider()

# ══════════════════════════════════════════════════════════════════
# SECTION 0 — System Overview
# ══════════════════════════════════════════════════════════════════

st.header("0 · System Overview")
st.markdown(
    "SciGraphAgent evaluates four retrieval conditions on multi-hop QA benchmarks. "
    "The **retry gate** (primary novelty) fires when answer quality falls below "
    "faithfulness and relevancy thresholds."
)

cols = st.columns(4)
condition_info = [
    ("A", "No Retrieval",      "LLM parametric memory only",        "Weakest baseline"),
    ("B", "Vector Only",       "ChromaDB cosine similarity (α=0.0)", "Standard RAG"),
    ("C", "Graph Only",        "BFS entity traversal (α=1.0)",       "Structural only"),
    ("D", "Hybrid Graph-RAG",  "Vector + Graph fusion (α=0.6)",      "★ Proposed system"),
]
for col, (letter, name, method, role) in zip(cols, condition_info):
    col.markdown(
        f"<div style='background:{PALETTE[letter]}22; "
        f"border-left:4px solid {PALETTE[letter]}; "
        f"padding:12px; border-radius:4px;'>"
        f"<b style='color:{PALETTE[letter]};'>Condition {letter}</b><br>"
        f"<b>{name}</b><br>"
        f"<small>{method}</small><br>"
        f"<small><i>{role}</i></small>"
        f"</div>",
        unsafe_allow_html=True,
    )

st.divider()

# ══════════════════════════════════════════════════════════════════
# SECTION 1 — Experiment 1: Retry Gate
# ══════════════════════════════════════════════════════════════════

st.header("1 · Experiment 1: Runtime RAGAS Retry Gate")
st.markdown(
    "**Primary novelty.** The gate evaluates faithfulness and relevancy after "
    "synthesis. If either score is below threshold, the agent re-retrieves with "
    "a larger top-k and regenerates. "
    "Compare to Self-RAG (Asai et al., ICLR 2024) which operates at token level."
)

e1 = metrics.get("exp1", {})
if e1:
    ng = e1.get("no_gate", {})
    wg = e1.get("with_gate", {})

    # Metric cards
    m1, m2, m3, m4, m5 = st.columns(5)
    delta_faith = wg.get("faithfulness", 0) - ng.get("faithfulness", 0)
    delta_f1    = wg.get("f1", 0) - ng.get("f1", 0)

    m1.metric("Faithfulness (no gate)",  f"{ng.get('faithfulness',0):.3f}")
    m2.metric("Faithfulness (w/ gate)",  f"{wg.get('faithfulness',0):.3f}",
              delta=f"{delta_faith:+.3f}")
    m3.metric("F1 (with gate)",          f"{wg.get('f1',0):.3f}",
              delta=f"{delta_f1:+.3f}")
    m4.metric("Gate trigger rate",
              f"{e1.get('gate_trigger_rate',0):.0%} "
              f"({e1.get('n_triggered',0)}/{e1.get('n_total',n)})")
    m5.metric("Latency overhead",
              f"+{e1.get('latency_overhead_s',0):.2f}s/query",
              help="Clock-measured — reliable even at n=3")

    # Plotly bar chart
    metrics_plot = ["Faithfulness", "F1 Score", "Relevancy"]
    keys_plot    = ["faithfulness",  "f1",       "relevancy"]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="No gate (single pass)",
        x=metrics_plot,
        y=[ng.get(k, 0) for k in keys_plot],
        marker_color=PALETTE["no_gate"],
        text=[f"{ng.get(k,0):.3f}" for k in keys_plot],
        textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name="With gate (RAGAS retry)",
        x=metrics_plot,
        y=[wg.get(k, 0) for k in keys_plot],
        marker_color=PALETTE["gate"],
        text=[f"{wg.get(k,0):.3f}" for k in keys_plot],
        textposition="outside",
    ))
    fig.add_hline(
        y=faith_thresh, line_dash="dot",
        line_color=PALETTE["thresh"],
        annotation_text=f"Faith threshold ({faith_thresh})",
        annotation_position="right",
    )
    fig.update_layout(
        title=f"Retry Gate vs Single Pass — {dataset.upper()} n={n}",
        barmode="group",
        yaxis=dict(range=[0, 1.2], title="Score (0–1)"),
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, width='stretch')

    st.info(
        f"**Self-RAG baseline (ICLR 2024):** HotpotQA F1 = 0.450 (token-level reflection).  \n"
        f"**Our gate F1 at n={n}:** {wg.get('f1',0):.3f}  \n"
        f"*Note: n={n} F1 is affected by retrieval coverage, not gate quality. "
        f"Gate effectiveness = latency overhead ({e1.get('latency_overhead_s',0):.2f}s) + "
        f"trigger rate ({e1.get('gate_trigger_rate',0):.0%}).*"
    )

st.divider()

# ══════════════════════════════════════════════════════════════════
# SECTION 2 — Experiment 2: Ablation
# ══════════════════════════════════════════════════════════════════

st.header("2 · Experiment 2: Four-Condition Retrieval Ablation")
st.markdown(
    f"**Target:** Context recall lift ≥ +{recall_target:.0%} for Condition D over B.  \n"
    "Motivated by Han et al. (2025), arXiv:2502.11371.  \n"
    "Validated by GraphRAG-R1 (WWW 2026): Text+Graph > Text only > Graph only."
)

e2 = metrics.get("exp2", {})
if e2:
    conds   = e2.get("per_condition", {})
    letters = ["A", "B", "C", "D"]

    # Build dataframe
    rows = []
    for letter, (cname, agg) in zip(letters, conds.items()):
        display = cname.split(": ", 1)[1] if ": " in cname[:3] else cname
        rows.append({
            "Condition":       f"{letter}: {display}",
            "Faithfulness":    round(agg.get("faithfulness", 0), 3),
            "F1 Score":        round(agg.get("f1", 0), 3),
            "Context Recall":  round(agg.get("context_recall", 0), 3),
            "Relevancy":       round(agg.get("relevancy", 0), 3),
        })
    df = pd.DataFrame(rows)

    # Recall lift
    b_recall = rows[1]["Context Recall"]
    d_recall = rows[3]["Context Recall"]
    lift     = d_recall - b_recall
    target_met = lift >= recall_target

    col_tab, col_lift = st.columns([3, 1])
    with col_tab:
        st.dataframe(
            df.set_index("Condition"),
            width='stretch',
        )
    with col_lift:
        st.metric(
            "Recall lift D over B",
            f"{lift:+.3f}",
            delta=f"Target: ≥ +{recall_target:.2f}",
            delta_color="normal" if target_met else "inverse",
        )
        if target_met:
            st.success("✓ Target met")
        else:
            st.warning(f"✗ Not met at n={n}\nExpected at n≥50")

    # Interactive grouped bar chart
    fig2 = go.Figure()
    metrics_ab = ["Faithfulness", "F1 Score", "Context Recall"]
    keys_ab    = ["faithfulness",  "f1",       "context_recall"]

    for letter, row in zip(letters, rows):
        fig2.add_trace(go.Bar(
            name=row["Condition"],
            x=metrics_ab,
            y=[row[m] for m in metrics_ab],
            marker_color=PALETTE[letter],
            text=[f"{row[m]:.3f}" for m in metrics_ab],
            textposition="outside",
        ))

    # Target recall line
    fig2.add_hline(
        y=b_recall + recall_target,
        line_dash="dash", line_color=PALETTE["thresh"],
        annotation_text=f"Recall target: B+{recall_target:.2f} = {b_recall+recall_target:.3f}",
        annotation_position="right",
    )
    fig2.update_layout(
        title=f"Four-Condition Ablation — {dataset.upper()} n={n}",
        barmode="group",
        yaxis=dict(range=[0, 1.25], title="Score (0–1)"),
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig2, width='stretch')

st.divider()

# ══════════════════════════════════════════════════════════════════
# SECTION 3 — Experiment 3: CI/CD Gate
# ══════════════════════════════════════════════════════════════════

st.header("3 · Experiment 3: CI/CD Faithfulness Regression Gate")
st.markdown(
    "**Engineering novelty.** Faithfulness threshold as a deployment gate. "
    "V1 (full config) should PASS; V2 (degraded: top-k=1, no graph) should be BLOCKED.  \n"
    f"*Reliable at n≥20 canonical questions. At n={n}, judge variance dominates.*"
)

e3 = metrics.get("exp3", {})
if e3:
    v1_faith  = e3.get("v1_faithfulness", 0.0)
    v2_faith  = e3.get("v2_faithfulness", 0.0)
    threshold = e3.get("threshold", 0.75)
    v1_pass   = v1_faith >= faith_thresh
    v2_block  = v2_faith <  faith_thresh

    col1, col2, col3 = st.columns(3)
    col1.metric("V1 Faithfulness (full)",    f"{v1_faith:.3f}",
                delta="PASS ✓" if v1_pass else "FAIL ❌")
    col2.metric("V2 Faithfulness (degraded)", f"{v2_faith:.3f}",
                delta="BLOCKED ✓" if v2_block else "PASSED ⚠")
    col3.metric("Gate threshold",             f"{faith_thresh:.2f}",
                help="Adjust in sidebar")

    # Gauge chart
    fig3 = go.Figure()
    configs = [
        ("V2: Degraded\n(top-k=1, no graph)", v2_faith,
         PALETTE["v2"] if not v2_block else "#F0E442"),
        ("V1: Full config\n(top-k=5, BFS α=0.6)", v1_faith,
         PALETTE["v1"] if v1_pass else PALETTE["v2"]),
    ]
    for i, (label, val, col) in enumerate(configs):
        fig3.add_trace(go.Bar(
            x=[val], y=[label],
            orientation="h",
            marker_color=col,
            text=[f"{val:.3f}"],
            textposition="outside",
            showlegend=False,
        ))
    fig3.add_vline(
        x=faith_thresh, line_dash="dash",
        line_color=PALETTE["thresh"], line_width=2.5,
        annotation_text=f"Threshold ({faith_thresh})",
        annotation_position="top right",
    )
    fig3.update_layout(
        title=f"CI/CD Regression Gate — {dataset.upper()} n={n}",
        xaxis=dict(range=[0, 1.2], title="Average Faithfulness Score"),
        height=300,
        barmode="overlay",
    )
    st.plotly_chart(fig3, width='stretch')

    gate_ok = v1_pass and v2_block
    if gate_ok:
        st.success("✓ Gate working correctly — V1 passed, V2 blocked")
    else:
        st.warning(
            f"Gate verdict at n={n}: V1 {'PASS' if v1_pass else 'FAIL'}, "
            f"V2 {'BLOCKED' if v2_block else 'PASSED (missed regression)'}.  \n"
            f"Judge variance at small n causes this. Reliable at n≥20 canonical questions."
        )

st.divider()

# ══════════════════════════════════════════════════════════════════
# SECTION 4 — Literature Comparison
# ══════════════════════════════════════════════════════════════════

st.header("4 · Literature Comparison")
st.markdown(
    "Positioning against published systems on HotpotQA F1.  \n"
    "*Note: Our F1 at n=3 is structurally 0 (retrieval coverage issue). "
    "Comparison valid at n=1000 with identical eval protocol.*"
)

e2_conds  = e2.get("per_condition", {}) if e2 else {}
d_cond    = list(e2_conds.values())[3] if len(e2_conds) >= 4 else {}
our_f1    = round(d_cond.get("f1", 0.0), 3) if d_cond else 0.0

lit_data = [
    {"System": "SciGraphAgent D (this work)",
     "HotpotQA F1": f"{our_f1} (n=3, pending n=1000)",
     "RAGAS Gate": "✓", "GPU-free": "✓", "Open Source": "✓",
     "Paper": "This work"},
    {"System": "Self-RAG",
     "HotpotQA F1": "0.450",
     "RAGAS Gate": "~", "GPU-free": "~", "Open Source": "✓",
     "Paper": "[arXiv:2310.11511](https://arxiv.org/abs/2310.11511)"},
    {"System": "HippoRAG 2",
     "HotpotQA F1": "~0.500",
     "RAGAS Gate": "✗", "GPU-free": "✗", "Open Source": "✓",
     "Paper": "[arXiv:2502.14802](https://arxiv.org/abs/2502.14802)"},
    {"System": "Graph-R1",
     "HotpotQA F1": "~0.580",
     "RAGAS Gate": "✗", "GPU-free": "✗", "Open Source": "✓",
     "Paper": "[arXiv:2507.21892](https://arxiv.org/abs/2507.21892)"},
    {"System": "GraphRAG-R1",
     "HotpotQA F1": "~0.620",
     "RAGAS Gate": "✗", "GPU-free": "✗", "Open Source": "✓",
     "Paper": "[arXiv:2507.23581](https://arxiv.org/abs/2507.23581)"},
    {"System": "MS GraphRAG",
     "HotpotQA F1": "N/A*",
     "RAGAS Gate": "✗", "GPU-free": "✗", "Open Source": "✓",
     "Paper": "[arXiv:2404.16130](https://arxiv.org/abs/2404.16130)"},
]

df_lit = pd.DataFrame(lit_data)
st.dataframe(df_lit.set_index("System"), width='stretch')
st.caption(
    "~ = partial | N/A* = not evaluated on this benchmark | "
    "Our unique combination: GPU-free + RAGAS gate + open source"
)

st.divider()

# ══════════════════════════════════════════════════════════════════
# SECTION 5 — Per-Question Drill-Down
# ══════════════════════════════════════════════════════════════════

st.header("5 · Per-Question Drill-Down")
st.markdown("Inspect individual question results from Experiment 2.")

if raw and "exp2" in raw:
    conds_raw = raw["exp2"].get("conditions", {})
    if conds_raw:
        cond_options = list(conds_raw.keys())
        selected_cond = st.selectbox("Select condition:", cond_options)

        results_for_cond = conds_raw.get(selected_cond, [])
        if results_for_cond:
            rows_drill = []
            for i, rec in enumerate(results_for_cond):
                rows_drill.append({
                    "Q#":           i + 1,
                    "Question":     rec.get("question", "")[:60] + "...",
                    "Gold Answer":  rec.get("gold", rec.get("answer", ""))[:30],
                    "Pred Answer":  rec.get("answer", "")[:30],
                    "Faithfulness": round(rec.get("faithfulness", 0), 3),
                    "Relevancy":    round(rec.get("relevancy", 0), 3),
                    "F1":           round(rec.get("f1", 0), 3),
                    "EM":           round(rec.get("em", 0), 3),
                    "Recall":       round(rec.get("context_recall", 0), 3),
                })
            df_drill = pd.DataFrame(rows_drill).set_index("Q#")
            st.dataframe(df_drill, width='stretch')

            # Highlight which questions have faith < threshold
            low_faith = [r for r in rows_drill if r["Faithfulness"] < faith_thresh]
            if low_faith:
                st.warning(
                    f"⚠ {len(low_faith)} question(s) with faithfulness < {faith_thresh} "
                    f"(gate would fire on these):"
                )
                for r in low_faith:
                    st.markdown(f"  - Q{r['Q#']}: {r['Question']}")
            else:
                st.success(
                    f"✓ All questions passed faith ≥ {faith_thresh} threshold for this condition"
                )
else:
    st.info("No raw results found. Run step03 first.")

st.divider()

# ── Footer ─────────────────────────────────────────────────────────
st.markdown(
    "<div style='text-align:center; color:gray; font-size:12px;'>"
    "SciGraphAgent Benchmark | GPU-free Graph-RAG with RAGAS Retry Gate | "
    "github.com/DrBala-2-0/scigraphagent-benchmark"
    "</div>",
    unsafe_allow_html=True,
)
