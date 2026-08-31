"""
step02_build_retrieval_systems.py
==================================
Builds two retrieval structures from the downloaded benchmark data:

  1. ChromaDB Vector Index
     - Splits each context passage into sentence-level chunks
     - Embeds each chunk using all-MiniLM-L6-v2 (384-dim, CPU-only)
     - Stores embeddings in a persistent ChromaDB collection
     - Enables cosine similarity search at query time

  2. In-Memory Knowledge Graph
     - Extracts capitalised entity pairs from each passage (co-occurrence)
     - Builds a directed graph: entity nodes, CO_OCCURS_WITH edges
     - Each edge carries the source record_id for provenance
     - Enables BFS traversal at query time

WHY TWO STRUCTURES?
-------------------
Han et al. (2025) — RAG vs GraphRAG (arXiv:2502.11371) — show that:
  - Dense vector search outperforms on single-hop factual queries
  - Graph traversal outperforms on multi-hop relational queries
  - Combining both outperforms either alone across all query types
This is the theoretical grounding for our alpha-fusion design (alpha=0.6).

RETRIEVAL COMPARISON vs COMPETING SYSTEMS
------------------------------------------
  SciGraphAgent (ours) : Dense vector (ChromaDB) + BFS graph — no BM25, no reranker
  Self-RAG             : Dense vector only (DPR-style) — no graph
  HippoRAG 2           : Dense vector + Personalised PageRank graph
  Graph-R1             : RL-trained agent over knowledge hypergraph (GPU required)
  GraphRAG-R1          : Hybrid graph-textual (RL-trained, GPU required)
  LightRAG             : Dual-level entity embedding + graph (no BM25)
  MS GraphRAG          : Graph traversal only (community + BFS, no vector)

None of the compared systems use BM25 or cross-encoder reranking on
HotpotQA/MuSiQue/2WikiMultiHop. Our combination is directly comparable.

WHY NOT BM25 OR RERANKING?
---------------------------
  BM25       : Would improve precision on exact entity names. Acknowledged
               gap. Adding BM25 would improve over ALL compared systems —
               it is a planned improvement, not a catch-up move.
  Reranking  : Cross-encoder rerankers add 0.5-2s per chunk on CPU.
               With 8GB RAM and a 5s latency target, borderline feasible
               but risky. Planned improvement for a later version.

WHY 384 DIMENSIONS?
-------------------
  all-MiniLM-L6-v2 produces 384-dim embeddings — a deliberate trade-off:
  - Larger (1536-dim Ada-002, 768-dim nomic-embed): better quality, more RAM
  - Smaller (128-dim): faster but loses semantic precision
  - 384-dim: fast on CPU (~10ms/chunk), fits thousands of chunks in 8GB RAM
  Planned upgrade: nomic-embed-text-v1.5 (768-dim, 8192-token context)
  which is used by GraphAgents and outperforms Ada-002 on technical text.

USAGE
-----
    python step02_build_retrieval_systems.py                    # default hotpotqa n=50
    python step02_build_retrieval_systems.py --dataset hotpotqa --n 3
    python step02_build_retrieval_systems.py --dataset musique  --n 3
"""

import json
import re
import argparse
import time
from pathlib import Path
from collections import defaultdict

DATA_DIR  = Path("data")
INDEX_DIR = Path("index")
INDEX_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════
# PART A — ChromaDB Vector Index
# ══════════════════════════════════════════════════════════════════

def build_vector_index(records: list[dict],
                       collection_name: str) -> object:
    """
    Embeds all context passages and stores them in ChromaDB.

    Embedding model : all-MiniLM-L6-v2 (384-dim, CPU-only)
    Similarity      : cosine distance
    Chunking        : sentence-level split (proxy for 512-token chunks)
    Storage         : persistent on disk at index/chroma/

    Why sentence-level chunking here?
    The full pipeline uses RecursiveCharacterTextSplitter at 512 tokens.
    For the benchmark we use sentence splits as a fast, dependency-free
    alternative that demonstrates the same retrieval principle.
    """
    import chromadb
    from chromadb.utils import embedding_functions

    print("  [A] Building ChromaDB vector index...")
    t0 = time.perf_counter()

    # Create persistent client — survives across Python sessions
    client = chromadb.PersistentClient(
        path=str(INDEX_DIR / "chroma")
    )

    # Embedding function — downloads model on first call (~80MB), cached after
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    # Delete existing collection so we start fresh each run
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        embedding_function=emb_fn,
        metadata={"hnsw:space": "cosine"}
    )

    # Split each record's context into sentence-level chunks
    documents, metadatas, ids = [], [], []

    for rec in records:
        # Sentence split — simple regex on .!? boundaries
        sentences = [
            s.strip()
            for s in re.split(r'(?<=[.!?])\s+', rec["context"])
            if len(s.strip()) > 30     # skip very short fragments
        ]

        for i, sent in enumerate(sentences[:12]):   # cap at 12 per record
            chunk_id = f"{rec['id']}_{i}"
            documents.append(sent)
            metadatas.append({
                "record_id": rec["id"],
                "dataset":   rec["dataset"],
                "answer":    rec["answer"][:80],
                "q_type":    rec["q_type"],
                "chunk_idx": i,
            })
            ids.append(chunk_id)

    # Batch insert — ChromaDB limit is 5461 per call
    BATCH = 500
    for start in range(0, len(documents), BATCH):
        collection.add(
            documents=documents[start:start + BATCH],
            metadatas=metadatas[start:start + BATCH],
            ids=ids[start:start + BATCH],
        )

    elapsed = time.perf_counter() - t0
    print(f"     Indexed  : {len(documents)} chunks from {len(records)} records")
    print(f"     Time     : {elapsed:.1f}s")
    print(f"     Model    : all-MiniLM-L6-v2 (384-dim, CPU)")
    print(f"     Storage  : {INDEX_DIR / 'chroma'}")

    return collection


def demo_vector_retrieval(collection, query: str,
                          top_k: int = 3) -> None:
    """Show what vector retrieval returns for a sample query."""
    print(f"\n  Vector retrieval demo — query: '{query[:60]}'")
    results = collection.query(
        query_texts=[query],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    )):
        similarity = 1 - dist
        print(f"  [{i+1}] sim={similarity:.3f} | {doc[:80]}...")
        print(f"       record_id={meta['record_id'][:20]} | answer={meta['answer'][:30]}")


# ══════════════════════════════════════════════════════════════════
# PART B — In-Memory Knowledge Graph
# ══════════════════════════════════════════════════════════════════

class KnowledgeGraph:
    """
    Lightweight in-memory knowledge graph built from entity co-occurrences.

    In the full SciGraphAgent pipeline, this graph is built by:
      Stage 1 — scispaCy NER (12 scientific entity types)
      Stage 2 — Claude relation extraction (12 typed relations)

    Here we use capitalised noun-phrase co-occurrence as a fast,
    dependency-free proxy that demonstrates the same retrieval principle.
    The graph structure and BFS traversal are identical to the full system.

    Nodes : entity strings (lowercased for deduplication)
    Edges : (entity_A) --CO_OCCURS_WITH--> (entity_B)
            each edge carries source record_id for provenance tracing

    Why co-occurrence rather than typed relations?
    Co-occurrence is extracted with a single regex — no API calls, no GPU.
    It is sufficient for the benchmark demonstration. The full pipeline
    replaces this with LLM-extracted typed triples (USES_METHOD,
    PROPOSES, EVALUATES, etc.) as described in Section 4.2 of the paper.
    """

    def __init__(self):
        # adjacency list: entity -> [(neighbour, record_id), ...]
        self.adj: dict[str, list[tuple[str, str]]] = defaultdict(list)
        self._node_set: set[str] = set()
        self._edge_count: int = 0

    def add_passage(self, text: str, record_id: str) -> None:
        """Extract entity pairs and add edges to the graph."""
        pairs = self._extract_entity_pairs(text)
        for a, b in pairs:
            a = a.lower().strip()
            b = b.lower().strip()
            if a == b or len(a) < 3 or len(b) < 3:
                continue
            self.adj[a].append((b, record_id))
            self.adj[b].append((a, record_id))
            self._node_set.update([a, b])
            self._edge_count += 1

    def _extract_entity_pairs(self,
                              text: str) -> list[tuple[str, str]]:
        """
        Extract capitalised noun phrases as candidate entities.
        Returns sequential pairs (co-occurrence within context window).

        In the full pipeline this is replaced by:
          scispaCy NER -> Claude relation extraction
        """
        STOPWORDS = {
            "The", "A", "An", "In", "On", "For", "Of", "To",
            "Is", "It", "We", "This", "That", "He", "She",
            "They", "His", "Her", "Its", "At", "By", "As",
        }
        pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b'
        entities = re.findall(pattern, text)
        entities = [
            e for e in dict.fromkeys(entities)   # deduplicate, preserve order
            if e not in STOPWORDS
        ]
        # Return sequential adjacent pairs
        return [
            (entities[i], entities[i + 1])
            for i in range(len(entities) - 1)
        ]

    def bfs(self, seeds: list[str],
            depth: int = 2,
            max_paths: int = 15) -> list[str]:
        """
        Breadth-first search from seed entities.

        Returns human-readable relation chains:
          'entity_a → CO_OCCURS_WITH → entity_b  [source: record_id]'

        This mirrors the BFS traversal in SciGraphAgent's
        GraphRAGEngine (retrieval/graph_rag.py).

        Planned upgrade: Semantic-Stop BFS (GraphAgents, Stewart et al.
        2026) — only accepts paths through a semantically meaningful
        waypoint node, making cross-domain discovery structured.
        """
        paths = []
        visited: set[str] = set()
        # queue: (current_node, path_so_far, current_depth)
        queue = [
            (s.lower(), [], 0)
            for s in seeds
            if s.lower() in self.adj
        ]

        while queue and len(paths) < max_paths:
            current, path_so_far, d = queue.pop(0)
            if current in visited or d > depth:
                continue
            visited.add(current)

            for neighbour, source_id in self.adj[current][:4]:
                if neighbour not in visited:
                    new_path = path_so_far + [current]
                    chain = " → CO_OCCURS_WITH → ".join(
                        new_path + [neighbour]
                    )
                    paths.append(
                        f"{chain}  [source: {source_id[:20]}]"
                    )
                    queue.append((neighbour, new_path, d + 1))

        return paths[:max_paths]

    def seed_entities(self, query: str) -> list[str]:
        """
        Find graph nodes that match words in the query.

        This is the seed extraction step of graph retrieval.
        Planned upgrade: embedding-based query-to-node mapping
        (GraphAgents, Stewart et al. 2026) — cosine similarity
        between query embedding and node embeddings replaces
        this regex heuristic, handling unseen entity phrasings.
        """
        # Match 4+ character words that appear as graph nodes
        words = re.findall(r'\b[a-z]{4,}\b', query.lower())
        seeds = [w for w in words if w in self.adj]

        # Also try capitalised phrases from the query
        caps = re.findall(
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,1})\b', query
        )
        seeds += [c.lower() for c in caps if c.lower() in self.adj]

        # Deduplicate, return top 5
        return list(dict.fromkeys(seeds))[:5]

    @property
    def node_count(self) -> int:
        return len(self._node_set)

    @property
    def edge_count(self) -> int:
        return self._edge_count

    @property
    def stats(self) -> dict:
        return {
            "nodes":            self.node_count,
            "edges":            self._edge_count,
            "density":          round(
                self._edge_count / max(self.node_count, 1), 2
            ),
            "avg_degree":       round(
                sum(len(v) for v in self.adj.values())
                / max(self.node_count, 1), 2
            ),
        }


def build_knowledge_graph(records: list[dict]) -> KnowledgeGraph:
    """Build the knowledge graph from all context passages."""
    print("  [B] Building in-memory knowledge graph...")
    t0 = time.perf_counter()

    kg = KnowledgeGraph()
    for rec in records:
        kg.add_passage(rec["context"], rec["id"])

    elapsed = time.perf_counter() - t0
    stats = kg.stats
    print(f"     Nodes    : {stats['nodes']:,}")
    print(f"     Edges    : {stats['edges']:,}")
    print(f"     Density  : {stats['density']} edges/node")
    print(f"     Time     : {elapsed:.2f}s")

    return kg


def demo_graph_retrieval(kg: KnowledgeGraph,
                         query: str) -> None:
    """Show what graph BFS returns for a sample query."""
    seeds = kg.seed_entities(query)
    paths = kg.bfs(seeds, depth=2, max_paths=5)
    print(f"\n  Graph retrieval demo — query: '{query[:60]}'")
    print(f"  Seed entities: {seeds}")
    if paths:
        print("  BFS paths found:")
        for p in paths:
            print(f"    {p}")
    else:
        print("  No graph paths found for this query")


# ══════════════════════════════════════════════════════════════════
# PART C — The Four Retrieval Conditions
# ══════════════════════════════════════════════════════════════════

class BaseRetriever:
    """Common interface for all four retrieval conditions."""
    name:  str = "base"
    alpha: float = 0.0

    def retrieve(self, query: str, top_k: int = 5) -> str:
        raise NotImplementedError


class ConditionA_NoRetrieval(BaseRetriever):
    """
    Condition A — No retrieval (LLM parametric knowledge only).
    Weakest baseline. Measures how much the LLM already knows
    from training without any external context.
    Returns empty string — the LLM must answer from memory alone.
    """
    name  = "A: No retrieval (LLM only)"
    alpha = 0.0

    def retrieve(self, query: str, top_k: int = 5) -> str:
        return ""


class ConditionB_VectorOnly(BaseRetriever):
    """
    Condition B — Vector-only RAG (alpha = 0.0).
    Standard RAG baseline: ChromaDB cosine similarity search.
    No graph traversal.

    This is the approach used by:
    - Self-RAG (Asai et al., ICLR 2024, arXiv:2310.11511)
    - Vanilla RAG baselines in HopRAG, CIRAG, BridgeRAG papers

    Context formula: 1.0 × vector_chunks + 0.0 × graph_paths
    """
    name  = "B: Vector-only RAG (alpha=0.0)"
    alpha = 0.0

    def __init__(self, collection):
        self.collection = collection

    def retrieve(self, query: str, top_k: int = 5) -> str:
        n = min(top_k, self.collection.count())
        if n == 0:
            return "[No documents indexed]"
        results = self.collection.query(
            query_texts=[query],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )
        lines = ["[Vector Retrieval Context]"]
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            sim = round(1 - dist, 3)
            lines.append(f"[sim={sim}] {doc}")
        return "\n".join(lines)


class ConditionC_GraphOnly(BaseRetriever):
    """
    Condition C — Graph-only BFS (alpha = 1.0).
    Knowledge graph traversal only. No vector similarity.

    Tests whether structural graph context alone is sufficient
    without semantic similarity grounding.

    Closest to:
    - MS GraphRAG local search (Edge et al., 2024, arXiv:2404.16130)
      which uses entity neighbourhood BFS without vector search

    Context formula: 0.0 × vector_chunks + 1.0 × graph_paths
    """
    name  = "C: Graph-only BFS (alpha=1.0)"
    alpha = 1.0

    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg

    def retrieve(self, query: str, top_k: int = 5) -> str:
        seeds = self.kg.seed_entities(query)
        paths = self.kg.bfs(seeds, depth=2, max_paths=top_k * 3)
        lines = ["[Graph Traversal Context]"]
        if seeds:
            lines.append(f"Seed entities: {seeds}")
            lines.extend(paths)
        else:
            lines.append("(No entity seeds found for this query)")
        return "\n".join(lines)


class ConditionD_HybridGraphRAG(BaseRetriever):
    """
    Condition D — Hybrid Graph-RAG (alpha = 0.6).
    The proposed SciGraphAgent retrieval engine.

    Combines:
      - Vector retrieval  (weight = 1 - alpha = 0.4)
      - Graph BFS         (weight = alpha       = 0.6)

    Alpha = 0.6 is graph-leaning, motivated by Han et al. (2025)
    showing graph retrieval outperforms on multi-hop questions.

    Closest to:
    - GraphRAG-R1 (Yu et al., WWW 2026, arXiv:2507.23581)
      which calls this "hybrid graph-textual retrieval" and
      shows Text+Graph > Text only > Graph only in ablation

    Our primary novelty over GraphRAG-R1:
      - No GPU required
      - Open source, pip-installable
      - Runtime RAGAS evaluation gate (their system has none)

    Context formula: (1-alpha) × vector_chunks + alpha × graph_paths
    """
    name  = "D: Hybrid Graph-RAG (alpha=0.6)"
    alpha = 0.6

    def __init__(self, collection, kg: KnowledgeGraph,
                 alpha: float = 0.6):
        self.collection = collection
        self.kg         = kg
        self.alpha      = alpha

    def retrieve(self, query: str, top_k: int = 5) -> str:
        # ── Vector branch ─────────────────────────────────────────
        vector_k = max(1, int(top_k * (1 - self.alpha)))
        n = min(vector_k + 2, self.collection.count())
        v_results = self.collection.query(
            query_texts=[query],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )
        v_chunks = v_results["documents"][0]
        v_scores = [
            round(1 - d, 3)
            for d in v_results["distances"][0]
        ]

        # ── Graph branch ──────────────────────────────────────────
        graph_k = max(1, int(top_k * self.alpha))
        seeds   = self.kg.seed_entities(query)
        g_paths = self.kg.bfs(
            seeds, depth=2, max_paths=graph_k * 3
        )

        # ── Assembly ──────────────────────────────────────────────
        lines = [
            f"[Hybrid Graph-RAG Context | alpha={self.alpha}]",
            "",
            f"── Vector Retrieval (weight {1-self.alpha:.1f}) ──",
        ]
        for chunk, score in zip(v_chunks, v_scores):
            lines.append(f"[sim={score}] {chunk}")

        lines.append("")
        lines.append(f"── Graph Traversal (weight {self.alpha:.1f}) ──")
        if seeds:
            lines.append(f"Seed entities: {seeds}")
            lines.extend(g_paths[:graph_k * 3])
        else:
            lines.append("(No graph seeds found for this query)")

        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main(dataset: str = "hotpotqa", n: int = 50) -> dict:
    print("\n" + "=" * 55)
    print("  STEP 02: BUILD RETRIEVAL SYSTEMS")
    print("=" * 55)
    print(f"  Dataset  : {dataset}")
    print(f"  N        : {n}")
    print(f"  Index dir: {INDEX_DIR.resolve()}\n")

    # ── Load data ─────────────────────────────────────────────────
    data_file = DATA_DIR / f"{dataset}_sample_{n}.json"
    if not data_file.exists():
        print(f"  ERROR: {data_file} not found.")
        print(f"  Run first: python step01_load_benchmarks.py --n {n}")
        return {}

    with open(data_file) as f:
        records = json.load(f)
    print(f"  Loaded {len(records)} records from {data_file.name}\n")

    # ── Build vector index ────────────────────────────────────────
    collection_name = f"benchmark_{dataset}_{n}"
    collection = build_vector_index(records, collection_name)

    # ── Build knowledge graph ─────────────────────────────────────
    print()
    kg = build_knowledge_graph(records)

    # ── Instantiate all four retrievers ───────────────────────────
    retrievers = [
        ConditionA_NoRetrieval(),
        ConditionB_VectorOnly(collection),
        ConditionC_GraphOnly(kg),
        ConditionD_HybridGraphRAG(collection, kg, alpha=0.6),
    ]

    # ── Smoke test — show retrieval output for question 1 ─────────
    print("\n" + "─" * 55)
    print("  SMOKE TEST — Retrieval preview on question 1")
    print("─" * 55)
    q = records[0]["question"]
    a = records[0]["answer"]
    print(f"  Question : {q}")
    print(f"  Answer   : {a}\n")

    demo_vector_retrieval(collection, q, top_k=2)
    demo_graph_retrieval(kg, q)

    print("\n" + "─" * 55)
    print("  ALL FOUR CONDITIONS — context preview")
    print("─" * 55)
    for ret in retrievers:
        ctx = ret.retrieve(q, top_k=3)
        preview = ctx[:180].replace("\n", " | ")
        print(f"\n  [{ret.name}]")
        print(f"    {preview}...")

    # ── Save config for downstream steps ──────────────────────────
    config = {
        "dataset":         dataset,
        "n_records":       len(records),
        "collection_name": collection_name,
        "alpha":           0.6,
        "kg_stats":        kg.stats,
        "retrievers": [
            {"id": r.name[0], "name": r.name, "alpha": r.alpha}
            for r in retrievers
        ],
    }
    config_path = DATA_DIR / "retrieval_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print("\n" + "=" * 55)
    print(f"  ✓ Vector index  : {collection.count()} chunks")
    print(f"  ✓ Knowledge graph: {kg.node_count:,} nodes, "
          f"{kg.edge_count:,} edges")
    print(f"  ✓ Config saved  : {config_path}")
    print("  ✓ Next: python step03_run_experiments.py")
    print("=" * 55)

    return {"collection": collection, "kg": kg,
            "retrievers": retrievers, "records": records}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build vector index and knowledge graph"
    )
    parser.add_argument(
        "--dataset", default="hotpotqa",
        choices=["hotpotqa", "musique", "2wikimultihopqa"]
    )
    parser.add_argument(
        "--n", type=int, default=50,
        help="Sample size matching step01 (default: 50)"
    )
    args = parser.parse_args()
    main(args.dataset, args.n)