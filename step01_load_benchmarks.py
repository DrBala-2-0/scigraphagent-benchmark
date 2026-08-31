"""
step01_load_benchmarks.py
=========================
Downloads and samples three standard multi-hop QA benchmark datasets
from HuggingFace. Saves each sample as a JSON file to ./data/

Datasets:
  - HotpotQA        (Yang et al., EMNLP 2018,  arXiv:1809.09600)
  - MuSiQue         (Trivedi et al., TACL 2022, arXiv:2108.00573)
  - 2WikiMultiHopQA (Ho et al., COLING 2020,   arXiv:2011.01060)

Usage:
  python step01_load_benchmarks.py           # default n=50
  python step01_load_benchmarks.py --n 3    # smoke test
  python step01_load_benchmarks.py --n 1000 # full protocol
"""

import json
import argparse
import random
from pathlib import Path
from datetime import datetime
from collections import Counter

# ── Constants ─────────────────────────────────────────────────────
DATA_DIR    = Path("data")
DATA_DIR.mkdir(exist_ok=True)
RANDOM_SEED = 42   # fixed seed — same shuffle every run = reproducible results


# ── Dataset loaders ───────────────────────────────────────────────

def load_hotpotqa(n: int) -> list[dict]:
    """
    HotpotQA — 97k multi-hop questions over Wikipedia.
    Each question requires bridging two Wikipedia passages.
    Reference: Yang et al., arXiv:1809.09600
    """
    print("  Downloading HotpotQA from HuggingFace...")
    from datasets import load_dataset

    ds = load_dataset(
        "hotpotqa/hotpot_qa", "distractor",
        split="validation",
    )
    sample = ds.shuffle(seed=RANDOM_SEED).select(range(min(n, len(ds))))

    records = []
    for row in sample:
        passages = []
        for title, sentences in zip(
            row["context"]["title"],
            row["context"]["sentences"]
        ):
            passages.append(f"[{title}] " + " ".join(sentences))

        records.append({
            "id":       row["id"],
            "question": row["question"],
            "answer":   row["answer"],
            "context":  "\n".join(passages[:10]),
            "dataset":  "hotpotqa",
            "q_type":   row.get("type", "unknown"),
        })
    return records


def load_musique(n: int) -> list[dict]:
    """
    MuSiQue — 25k compositional multi-hop questions (2-4 hops).
    Reference: Trivedi et al., arXiv:2108.00573
    """
    print("  Downloading MuSiQue from HuggingFace...")
    from datasets import load_dataset

    ds = load_dataset(
        "bdsaglam/musique",
        split="validation",
    )
    sample = ds.shuffle(seed=RANDOM_SEED).select(range(min(n, len(ds))))

    records = []
    for row in sample:
        supporting = [
            p["paragraph_text"]
            for p in row["paragraphs"]
            if p.get("is_supporting", False)
        ]
        all_paras = [p["paragraph_text"] for p in row["paragraphs"]]
        context = (
            "\n".join(supporting[:6])
            if supporting
            else "\n".join(all_paras[:6])
        )
        records.append({
            "id":       row["id"],
            "question": row["question"],
            "answer":   row["answer"],
            "context":  context,
            "dataset":  "musique",
            "q_type":   f"{len(row.get('question_decomposition', []))}-hop",
        })
    return records


def load_2wikimultihopqa(n: int) -> list[dict]:
    """
    2WikiMultiHopQA — 192k relation-chaining questions.
    Reference: Ho et al., arXiv:2011.01060
    """
    print("  Downloading 2WikiMultiHopQA from HuggingFace...")
    from datasets import load_dataset

    ds = load_dataset(
        "framolfese/2WikiMultihopQA",
        split="validation",
    )
    sample = ds.shuffle(seed=RANDOM_SEED).select(range(min(n, len(ds))))

    records = []
    for row in sample:
        ctx_raw = row.get("context", "")

        # framolfese/2WikiMultihopQA stores context in HotpotQA format:
        # {"title": [...], "sentences": [[...], [...]]}
        if isinstance(ctx_raw, dict) and "title" in ctx_raw and "sentences" in ctx_raw:
            passages = []
            for title, sents in zip(ctx_raw["title"], ctx_raw["sentences"]):
                passages.append(f"[{title}] " + " ".join(sents))
            context = "\n".join(passages[:10])
        elif isinstance(ctx_raw, list):
            context = "\n".join(str(p) for p in ctx_raw[:8])
        else:
            context = str(ctx_raw)

        records.append({
            "id":       row.get("id", ""),
            "question": row["question"],
            "answer":   row["answer"],
            "context":  context,
            "dataset":  "2wikimultihopqa",
            "q_type":   row.get("type", "unknown"),
        })
    return records


# ── Summary printer ───────────────────────────────────────────────

def print_summary(records: list[dict], name: str) -> None:
    types = Counter(r["q_type"] for r in records)
    print(f"\n  [{name}] {len(records)} questions sampled")
    for qtype, count in types.most_common():
        print(f"    {qtype:30s}: {count}")
    # Show one example question
    if records:
        print(f"\n  Example question:")
        print(f"    Q: {records[0]['question']}")
        print(f"    A: {records[0]['answer']}")


# ── Main ──────────────────────────────────────────────────────────

def main(n: int) -> None:
    print("\n" + "=" * 55)
    print("  STEP 01: LOAD BENCHMARK DATASETS")
    print("=" * 55)
    print(f"  Sample size : {n} questions per dataset")
    print(f"  Output dir  : {DATA_DIR.resolve()}")
    print(f"  Random seed : {RANDOM_SEED} (reproducible)\n")

    loaders = [
        ("hotpotqa",        load_hotpotqa),
        ("musique",         load_musique),
        ("2wikimultihopqa", load_2wikimultihopqa),
    ]

    saved = {}
    for name, loader in loaders:
        print(f"\n[{name.upper()}]")
        try:
            records = loader(n)
            out_path = DATA_DIR / f"{name}_sample_{n}.json"
            with open(out_path, "w") as f:
                json.dump(records, f, indent=2)
            print_summary(records, name)
            print(f"\n  Saved → {out_path}")
            saved[name] = str(out_path)
        except Exception as e:
            print(f"  ERROR loading {name}: {e}")
            print("  Skipping — continuing with other datasets")

    # Write manifest — downstream steps use this to find data files
    manifest = {
        "created":  datetime.now().isoformat(),
        "sample_n": n,
        "seed":     RANDOM_SEED,
        "datasets": saved,
    }
    manifest_path = DATA_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 55)
    print(f"  ✓ {len(saved)}/3 datasets saved")
    print(f"  ✓ Manifest → {manifest_path}")
    print("  ✓ Next: python step02_build_retrieval_systems.py")
    print("=" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and sample multi-hop QA benchmark datasets"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=50,
        help="Questions to sample per dataset (default: 50)"
    )
    args = parser.parse_args()
    main(args.n)