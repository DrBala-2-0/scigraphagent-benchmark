"""
step05_visualise.py
====================
Generates five publication-quality figures from aggregated metrics.

NO API CALLS — reads results/metrics_{dataset}_{n}.json from step04.
Output: results/figures/*.png at 300 dpi

FIVE FIGURES
------------
Fig 1: Four-condition ablation bar chart
        Faithfulness, F1, Context Recall across A, B, C, D
        Target line at +15pp for recall lift

Fig 2: Retry gate paired bar chart
        No-gate vs with-gate: faithfulness, F1, relevancy
        Annotated with gate trigger rate and latency overhead

Fig 3: CI/CD regression gate gauge
        Horizontal bars for V1 and V2 against 0.75 threshold
        Clear PASS/BLOCKED verdict labels

Fig 4: Literature positioning scatter
        Our system vs published baselines on F1 axis
        Annotated with novelty axis markers

Fig 5: Per-question faithfulness heatmap
        Conditions × questions matrix
        Shows which questions benefit most from graph retrieval

DESIGN DECISIONS
----------------
- matplotlib only — no plotly, no seaborn dependency
- 300 dpi for paper submission
- Colour-blind safe palette (Wong 2011)
- All figures saved to results/figures/
- Each figure also displayed inline

USAGE
-----
    python step05_visualise.py
    python step05_visualise.py --dataset hotpotqa --n 3
"""

import json
import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for all environments
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Colour-blind safe palette (Wong 2011, Nature Methods) ──────────
# 8 colours distinguishable by people with colour vision deficiency
PALETTE = {
    "black":  "#000000",
    "orange": "#E69F00",
    "sky":    "#56B4E9",
    "green":  "#009E73",
    "yellow": "#F0E442",
    "blue":   "#0072B2",
    "red":    "#D55E00",
    "pink":   "#CC79A7",
}

# Condition colours — consistent across all figures
COND_COLOURS = {
    "A": PALETTE["black"],
    "B": PALETTE["sky"],
    "C": PALETTE["orange"],
    "D": PALETTE["blue"],
}


def set_style() -> None:
    """Apply consistent matplotlib style for all figures."""
    plt.rcParams.update({
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
        "axes.grid":         True,
        "grid.alpha":        0.3,
        "grid.linestyle":    "--",
        "font.family":       "sans-serif",
        "font.size":         11,
        "axes.titlesize":    13,
        "axes.labelsize":    11,
        "xtick.labelsize":   10,
        "ytick.labelsize":   10,
        "legend.fontsize":   10,
        "legend.framealpha": 0.9,
    })


def save_figure(fig: plt.Figure, name: str, dpi: int = 300) -> Path:
    """Save a figure to results/figures/ at the specified dpi."""
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"  Saved → {path} ({path.stat().st_size // 1024} KB)")
    return path


# ══════════════════════════════════════════════════════════════════
# FIGURE 1 — Four-condition ablation bar chart
# ══════════════════════════════════════════════════════════════════

def fig1_ablation(metrics: dict, dataset: str, n: int) -> plt.Figure:
    """
    Four-condition ablation: A, B, C, D across three metrics.

    Design decisions:
    - Grouped bars: conditions side by side for each metric
    - Target line at +0.15 recall lift on the recall panel
    - Error bars would require n≥2 per condition per metric
      (omitted at n=3; shown at n≥50 with std from aggregate)
    - Colour-blind safe palette — conditions A/B/C/D use
      black/sky/orange/blue consistently across all figures
    """
    e2 = metrics.get("exp2", {})
    conds = e2.get("per_condition", {})
    if not conds:
        print("  [Fig 1] No Experiment 2 data — skipping")
        return None

    letters  = ["A", "B", "C", "D"]
    metrics_to_plot = [
        ("faithfulness",    "Faithfulness"),
        ("f1",              "F1 Score"),
        ("context_recall",  "Context Recall"),
    ]

    # Extract values
    values = {m: [] for m, _ in metrics_to_plot}
    short_names = []
    for letter, (cname, agg) in zip(letters, conds.items()):
        display = cname.split(": ", 1)[1] if ": " in cname[:3] else cname
        # Shorten for x-axis
        short = {
            "No retrieval (LLM only)":      "A: None",
            "Vector-only RAG (alpha=0.0)":  "B: Vector",
            "Graph-only BFS (alpha=1.0)":   "C: Graph",
            "Hybrid Graph-RAG (alpha=0.6)": "D: Hybrid",
        }.get(display, f"{letter}: {display[:10]}")
        short_names.append(short)
        for m, _ in metrics_to_plot:
            values[m].append(agg.get(m, 0.0))

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(
        f"Fig 1: Four-Condition Retrieval Ablation\n"
        f"({dataset.upper()}, n={n})",
        fontsize=14, fontweight="bold", y=1.02
    )

    colours = [COND_COLOURS[l] for l in letters]
    x = np.arange(len(letters))
    bar_w = 0.6

    for ax, (metric, label) in zip(axes, metrics_to_plot):
        vals = values[metric]
        bars = ax.bar(x, vals, width=bar_w, color=colours,
                      edgecolor="white", linewidth=0.5)

        # Value labels on bars
        for bar, val in zip(bars, vals):
            if val > 0.01:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"{val:.3f}", ha="center", va="bottom",
                        fontsize=9, fontweight="bold")

        # Target line for recall
        if metric == "context_recall":
            b_val = vals[1]   # Condition B
            target = b_val + 0.15
            ax.axhline(target, color=PALETTE["red"], linestyle="--",
                       linewidth=1.5, alpha=0.8,
                       label=f"Target (+15pp over B) = {target:.2f}")
            ax.legend(fontsize=8)

        ax.set_title(label, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(short_names, fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Score (0–1)")
        ax.axhline(0, color="black", linewidth=0.5)

    # Shared legend for conditions
    patches = [mpatches.Patch(color=COND_COLOURS[l], label=sn)
               for l, sn in zip(letters, short_names)]
    fig.legend(handles=patches, loc="lower center",
               ncol=4, bbox_to_anchor=(0.5, -0.05),
               framealpha=0.9)

    note = (f"Note: n={n} smoke test. Target recall lift (+15pp) "
            f"detectable at n≥50.")
    fig.text(0.5, -0.12, note, ha="center", fontsize=9,
             color="gray", style="italic")

    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════
# FIGURE 2 — Retry gate paired bar chart
# ══════════════════════════════════════════════════════════════════

def fig2_retry_gate(metrics: dict, dataset: str, n: int) -> plt.Figure:
    """
    Retry gate: no-gate vs with-gate across faithfulness, F1, relevancy.

    Paired bars: left=no-gate, right=with-gate for each metric.
    Annotation box shows gate trigger rate and latency overhead —
    two metrics that are meaningful even at n=3.

    The latency overhead is the most reliable number in this figure
    because it is measured by the system clock, not the LLM judge.
    """
    e1 = metrics.get("exp1", {})
    if not e1:
        print("  [Fig 2] No Experiment 1 data — skipping")
        return None

    ng = e1.get("no_gate", {})
    wg = e1.get("with_gate", {})

    metric_labels = ["Faithfulness", "F1 Score", "Relevancy"]
    metric_keys   = ["faithfulness", "f1",        "relevancy"]

    ng_vals = [ng.get(k, 0.0) for k in metric_keys]
    wg_vals = [wg.get(k, 0.0) for k in metric_keys]

    x     = np.arange(len(metric_labels))
    bar_w = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(
        f"Fig 2: Runtime RAGAS Retry Gate — No-Gate vs With-Gate\n"
        f"({dataset.upper()}, n={n})",
        fontsize=14, fontweight="bold"
    )

    bars_ng = ax.bar(x - bar_w / 2, ng_vals, bar_w,
                     label="No gate (single pass)",
                     color=PALETTE["sky"], edgecolor="white")
    bars_wg = ax.bar(x + bar_w / 2, wg_vals, bar_w,
                     label="With gate (RAGAS retry)",
                     color=PALETTE["blue"], edgecolor="white")

    # Delta annotations
    for i, (nv, wv) in enumerate(zip(ng_vals, wg_vals)):
        delta = wv - nv
        if abs(delta) > 0.005:
            arrow_col = PALETTE["green"] if delta > 0 else PALETTE["red"]
            ax.annotate(
                f"Δ{delta:+.3f}",
                xy=(i + bar_w / 2, wv + 0.02),
                fontsize=9, color=arrow_col, fontweight="bold", ha="center"
            )
        # Value labels
        for bar, val in [(bars_ng[i], nv), (bars_wg[i], wv)]:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    val + 0.01, f"{val:.3f}",
                    ha="center", va="bottom", fontsize=8)

    # Info box — reliable metrics even at n=3
    gate_rate = e1.get("gate_trigger_rate", 0.0)
    latency   = e1.get("latency_overhead_s", 0.0)
    n_trig    = e1.get("n_triggered", 0)
    n_total   = e1.get("n_total", n)
    info_text = (
        f"Gate trigger rate: {gate_rate:.0%} ({n_trig}/{n_total})\n"
        f"Latency overhead: {latency:+.2f}s/query\n"
        f"(reliable even at n=3)"
    )
    ax.text(0.98, 0.97, info_text, transform=ax.transAxes,
            fontsize=9, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor=PALETTE["yellow"], alpha=0.8))

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Score (0–1)")
    ax.legend(loc="upper left")

    # Threshold line for faithfulness
    ax.axhline(0.75, color=PALETTE["red"], linestyle=":",
               linewidth=1.2, alpha=0.7,
               label="Faithfulness gate threshold (0.75)")

    note = (f"Note: n={n} — Δ values unreliable at small n. "
            f"Latency overhead is clock-measured and reliable.")
    ax.text(0.5, -0.08, note, transform=ax.transAxes,
            ha="center", fontsize=9, color="gray", style="italic")

    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════
# FIGURE 3 — CI/CD regression gate gauge
# ══════════════════════════════════════════════════════════════════

def fig3_cicd_gate(metrics: dict, dataset: str, n: int) -> plt.Figure:
    """
    CI/CD gate: horizontal bar chart comparing V1 and V2 faithfulness
    against the 0.75 deployment threshold.

    Horizontal bars are more intuitive than vertical for threshold
    comparisons — the threshold line is a vertical rule that the bar
    must cross to pass, which reads naturally left-to-right.
    """
    e3 = metrics.get("exp3", {})
    if not e3:
        print("  [Fig 3] No Experiment 3 data — skipping")
        return None

    v1_faith  = e3.get("v1_faithfulness", 0.0)
    v2_faith  = e3.get("v2_faithfulness", 0.0)
    threshold = e3.get("threshold", 0.75)
    v1_pass   = e3.get("v1_passed", False)
    v2_block  = e3.get("v2_blocked", False)

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.suptitle(
        f"Fig 3: CI/CD Faithfulness Regression Gate\n"
        f"({dataset.upper()}, n={n} — reliable at n≥20)",
        fontsize=14, fontweight="bold"
    )

    configs = [
        ("V2: Degraded\n(top-k=1, no graph)", v2_faith,
         PALETTE["red"] if not v2_block else PALETTE["orange"]),
        ("V1: Full config\n(top-k=5, BFS α=0.6)", v1_faith,
         PALETTE["green"] if v1_pass else PALETTE["orange"]),
    ]

    y_pos = [0, 1]
    for (label, val, col), y in zip(configs, y_pos):
        ax.barh(y, val, height=0.5, color=col, alpha=0.85,
                edgecolor="white")
        # Value label
        ax.text(val + 0.01, y, f"{val:.3f}",
                va="center", fontsize=11, fontweight="bold")
        # Verdict label
        if y == 1:
            verdict = "PASS ✓" if v1_pass else "FAIL ❌"
            ax.text(0.02, y, verdict, va="center", fontsize=10,
                    color="white", fontweight="bold")
        else:
            verdict = "BLOCKED ✓" if v2_block else "PASSED ⚠"
            ax.text(0.02, y, verdict, va="center", fontsize=10,
                    color="white", fontweight="bold")

    # Threshold line
    ax.axvline(threshold, color=PALETTE["red"], linewidth=2.5,
               linestyle="--", label=f"Threshold = {threshold}")
    ax.text(threshold + 0.005, 1.55,
            f"Threshold\n{threshold}", fontsize=9,
            color=PALETTE["red"], fontweight="bold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([c[0] for c in configs])
    ax.set_xlim(0, 1.2)
    ax.set_xlabel("Average Faithfulness Score")

    # Expected behaviour note
    gate_ok = v1_pass and v2_block
    status  = "Gate working correctly ✓" if gate_ok else \
              f"Gate unreliable at n={n} — needs n≥20 canonical questions"
    ax.text(0.5, -0.18, status, transform=ax.transAxes,
            ha="center", fontsize=10,
            color=PALETTE["green"] if gate_ok else PALETTE["red"],
            fontweight="bold")

    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════
# FIGURE 4 — Literature positioning scatter
# ══════════════════════════════════════════════════════════════════

def fig4_literature(metrics: dict, dataset: str, n: int) -> plt.Figure:
    """
    Position SciGraphAgent against published baselines.

    X-axis: HotpotQA F1 (published numbers from primary papers)
    Y-axis: System capabilities score (GPU-free=1pt, Gate=1pt, OSS=1pt)
    Marker size: proportional to capability score

    This is an honest positioning figure — our F1 at n=3 is shown
    as a hollow circle (pending n=1000 confirmation) to distinguish
    it from the solid markers of published systems.

    Note: The Y-axis (capability score) is a composite metric we
    defined to visualise the novelty axis. It is not a standard
    published metric — this is clearly labelled.
    """
    # Literature data — from primary papers
    systems = [
        # (name,              f1,    gpu_free, gate, oss, marker)
        ("SciGraphAgent\n(ours, n=3)", 0.000, 1, 1, 1, "o"),
        ("Self-RAG\n(ICLR 2024)",      0.450, 1, 0, 1, "s"),
        ("HippoRAG 2\n(ICML 2025)",    0.500, 0, 0, 1, "^"),
        ("Graph-R1\n(ICML 2026)",      0.580, 0, 0, 1, "D"),
        ("GraphRAG-R1\n(WWW 2026)",    0.620, 0, 0, 1, "P"),
    ]

    # Our actual F1 from metrics (may differ from 0.000 at n=50+)
    e2 = metrics.get("exp2", {})
    conds = e2.get("per_condition", {})
    if conds:
        d_cond = list(conds.values())[3]   # Condition D
        our_f1 = d_cond.get("f1", 0.0)
        systems[0] = ("SciGraphAgent\n(ours, n=3)", our_f1, 1, 1, 1, "o")

    fig, ax = plt.subplots(figsize=(11, 7))
    fig.suptitle(
        f"Fig 4: Literature Positioning — HotpotQA F1\n"
        f"({dataset.upper()}, n={n})",
        fontsize=14, fontweight="bold"
    )

    marker_colours = [
        PALETTE["blue"],   # ours
        PALETTE["sky"],    # Self-RAG
        PALETTE["green"],  # HippoRAG
        PALETTE["orange"], # Graph-R1
        PALETTE["red"],    # GraphRAG-R1
    ]

    for (name, f1, gpu_free, gate, oss, marker), col in \
            zip(systems, marker_colours):
        capability = gpu_free + gate + oss   # 0-3 composite
        size       = 200 + capability * 100

        # Our system: hollow marker (n=3 pending confirmation)
        if "ours" in name:
            ax.scatter(f1, capability, s=size, marker=marker,
                       facecolors="none", edgecolors=col,
                       linewidths=2.5, zorder=5)
        else:
            ax.scatter(f1, capability, s=size, marker=marker,
                       color=col, zorder=5, alpha=0.85)

        # Labels
        offsets = {
            "Graph-R1":    0.12,
            "GraphRAG-R1": -0.12,   # push one below
        }
        offset_y = 0.08 if "ours" in name else offsets.get(name.split("\n")[0], 0.06)
        ax.annotate(name, (f1, capability),
                    xytext=(f1, capability + offset_y),
                    fontsize=8.5, ha="center",
                    fontweight="bold" if "ours" in name else "normal")

    ax.set_xlabel("HotpotQA F1 Score (from primary papers)", fontsize=11)
    ax.set_ylabel("Capability Score\n(GPU-free + Gate + OSS, max=3)",
                  fontsize=11)
    ax.set_xlim(-0.1, 0.8)
    ax.set_ylim(-0.5, 4.0)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(["0\n(None)", "1\n(One)", "2\n(Two)", "3\n(All three)"])

    # Highlight our novelty region
    ax.axhspan(2.5, 3.5, alpha=0.05, color=PALETTE["blue"],
               label="Our novelty region (all 3 capabilities)")
    ax.legend(fontsize=9, loc="lower right")

    note = (
        "Hollow marker = our result at n=3 (pending n=1000 confirmation).\n"
        "Capability score is a composite metric defined for this figure. "
        "F1 numbers from primary paper abstracts/tables."
    )
    ax.text(0.01, -0.15, note, transform=ax.transAxes,
            fontsize=8, color="gray", style="italic")

    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════
# FIGURE 5 — Per-question faithfulness heatmap
# ══════════════════════════════════════════════════════════════════

def fig5_heatmap(metrics: dict, raw_results: dict,
                 dataset: str, n: int) -> plt.Figure:
    """
    Heatmap: conditions (rows) × questions (columns), value=faithfulness.

    Shows which questions benefit most from graph retrieval.
    Columns where C or D have higher faithfulness than B indicate
    questions where graph traversal found supporting context that
    vector search missed.

    Uses the raw results (not aggregated) to show per-question detail.
    This figure is only meaningful at n≥10 — at n=3, show it but
    note the limitation clearly.
    """
    exp2 = raw_results.get("exp2", {})
    conds = exp2.get("conditions", {})
    if not conds:
        print("  [Fig 5] No Experiment 2 data — skipping")
        return None

    letters      = ["A", "B", "C", "D"]
    cond_names   = list(conds.keys())
    n_questions  = max(len(v) for v in conds.values())
    n_conditions = len(conds)

    # Build matrix: rows=conditions, cols=questions
    matrix = np.zeros((n_conditions, n_questions))
    for i, (cname, results) in enumerate(conds.items()):
        for j, rec in enumerate(results):
            matrix[i, j] = rec.get("faithfulness", 0.0)

    # Short question labels
    q_labels = []
    first_cond = list(conds.values())[0]
    for j, rec in enumerate(first_cond):
        q = rec.get("question", f"Q{j+1}")
        q_labels.append(f"Q{j+1}: {q[:30]}...")

    # Condition row labels
    row_labels = []
    for letter, cname in zip(letters, cond_names):
        display = cname.split(": ", 1)[1] if ": " in cname[:3] else cname
        short   = {
            "No retrieval (LLM only)":      "A: None",
            "Vector-only RAG (alpha=0.0)":  "B: Vector",
            "Graph-only BFS (alpha=1.0)":   "C: Graph",
            "Hybrid Graph-RAG (alpha=0.6)": "D: Hybrid",
        }.get(display, f"{letter}: {display[:12]}")
        row_labels.append(short)

    fig, ax = plt.subplots(figsize=(max(8, n_questions * 2.5), 5))
    fig.suptitle(
        f"Fig 5: Per-Question Faithfulness Heatmap\n"
        f"({dataset.upper()}, n={n})",
        fontsize=14, fontweight="bold"
    )

    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto",
                   vmin=0.0, vmax=1.0)

    # Annotations in each cell
    for i in range(n_conditions):
        for j in range(n_questions):
            val = matrix[i, j]
            text_col = "black" if 0.3 < val < 0.7 else "white"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=10, fontweight="bold", color=text_col)

    ax.set_xticks(range(n_questions))
    ax.set_xticklabels(q_labels, rotation=15, ha="right", fontsize=9)
    ax.set_yticks(range(n_conditions))
    ax.set_yticklabels(row_labels, fontsize=10)
    ax.set_xlabel("Question", fontsize=11)
    ax.set_ylabel("Retrieval Condition", fontsize=11)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Faithfulness Score", fontsize=10)
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])

    note = (
        f"Green = high faithfulness (answer grounded in context). "
        f"Red = low. n={n} — heatmap meaningful at n≥10."
    )
    ax.text(0.5, -0.35, note, transform=ax.transAxes,
            ha="center", fontsize=9, color="gray", style="italic")
    plt.subplots_adjust(bottom=0.3)
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main(dataset: str = "hotpotqa", n: int = 50) -> None:
    print("\n" + "=" * 55)
    print("  STEP 05: VISUALISE RESULTS")
    print("=" * 55)
    print(f"  Dataset : {dataset}")
    print(f"  N       : {n}")
    print(f"  Output  : {FIGURES_DIR.resolve()}")
    print(f"  API calls: 0 | Cost: $0.00\n")

    # Load aggregated metrics
    metrics_path = RESULTS_DIR / f"metrics_{dataset}_{n}.json"
    if not metrics_path.exists():
        print(f"  ERROR: {metrics_path} not found.")
        print(f"  Run first: python step04_compute_metrics.py "
              f"--dataset {dataset} --n {n}")
        return

    with open(metrics_path) as f:
        metrics = json.load(f)

    # Load raw results for heatmap (needs per-question data)
    raw_path = RESULTS_DIR / f"raw_results_{dataset}_{n}.json"
    raw = {}
    if raw_path.exists():
        with open(raw_path) as f:
            raw = json.load(f)

    set_style()

    # Generate all five figures
    figures = [
        ("fig1_ablation",    fig1_ablation(metrics, dataset, n)),
        ("fig2_retry_gate",  fig2_retry_gate(metrics, dataset, n)),
        ("fig3_cicd_gate",   fig3_cicd_gate(metrics, dataset, n)),
        ("fig4_literature",  fig4_literature(metrics, dataset, n)),
        ("fig5_heatmap",     fig5_heatmap(metrics, raw, dataset, n)),
    ]

    print("  Saving figures...")
    saved = []
    for name, fig in figures:
        if fig is not None:
            path = save_figure(fig, f"{name}_{dataset}_{n}")
            saved.append(path)
            plt.close(fig)

    print()
    print("=" * 55)
    print(f"  ✓ {len(saved)}/5 figures saved to {FIGURES_DIR}")
    print("  ✓ Next: python step06_dashboard.py")
    print("=" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate publication figures from metrics"
    )
    parser.add_argument(
        "--dataset", default="hotpotqa",
        choices=["hotpotqa", "musique", "2wikimultihopqa"]
    )
    parser.add_argument(
        "--n", type=int, default=50,
        help="Sample size matching step04 run (default: 50)"
    )
    args = parser.parse_args()
    main(args.dataset, args.n)