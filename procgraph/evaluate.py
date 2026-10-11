"""Evaluate the three parts of the pipeline.

1. Retrieval: does the context contain the facts needed to answer? (context
   recall, GraphRAG vs. text-chunk RAG at the same context size)
2. Capability matching: does the matcher link requirements to the AI
   capabilities an expert would choose?
3. Extraction (optional): how close is an LLM extraction to the reference
   annotation? (precision / recall of entities and relations)
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import yaml

from .build import load_capabilities, normalise
from .metamodel import DATA_DIR
from .retrieve import ChunkRetriever, GraphRetriever


def load_evaluation() -> dict:
    return yaml.safe_load((DATA_DIR / "evaluation.yaml").read_text(encoding="utf-8"))


def corpus_paths() -> list[Path]:
    """Documents plus the capability taxonomy: the same knowledge the graph is built from."""
    return sorted((DATA_DIR / "documents").glob("*.md")) + [DATA_DIR / "ai_capabilities.yaml"]


def recall(context: str, expected: list[str]) -> float:
    text = context.lower()
    return sum(e.lower() in text for e in expected) / len(expected)


def corpus_chars() -> int:
    return sum(len(p.read_text(encoding="utf-8")) for p in corpus_paths())


def evaluate_retrieval(g: nx.MultiDiGraph, questions: list[dict], **retriever_kwargs) -> list[dict]:
    graph_retriever = GraphRetriever(g, **retriever_kwargs)
    chunk_retriever = ChunkRetriever(corpus_paths())
    rows = []
    for q in questions:
        graph_ctx = graph_retriever.retrieve(q["question"]).context
        # Give the baseline the same amount of context text as the graph retriever.
        chunk_ctx = chunk_retriever.retrieve(q["question"], char_budget=len(graph_ctx)).context
        rows.append({
            "question": q["question"],
            "hops": q["hops"],
            "graph": recall(graph_ctx, q["expected"]),
            "chunks": recall(chunk_ctx, q["expected"]),
            "chars": len(graph_ctx),
            "missing_graph": [e for e in q["expected"] if e.lower() not in graph_ctx.lower()],
            "missing_chunks": [e for e in q["expected"] if e.lower() not in chunk_ctx.lower()],
        })
    return rows


def mean(rows: list[dict], key: str, hops: str | None = None) -> float:
    subset = [r for r in rows if hops is None or r["hops"] == hops]
    return sum(r[key] for r in subset) / len(subset)


def sweep(g: nx.MultiDiGraph, questions: list[dict]) -> list[dict]:
    """Ablation: schema-aware boost on/off and context size (number of graph nodes)."""
    results = []
    for boost in (1.0, 3.0):
        for top_n in (20, 25, 30):
            rows = evaluate_retrieval(g, questions, type_boost=boost, top_n=top_n)
            results.append({
                "type_boost": boost,
                "top_n": top_n,
                "chars": round(mean(rows, "chars")),
                "graph_multi": mean(rows, "graph", "multi"),
                "chunks_multi": mean(rows, "chunks", "multi"),
                "graph_all": mean(rows, "graph"),
                "chunks_all": mean(rows, "chunks"),
            })
    return results


def evaluate_matching(g: nx.MultiDiGraph, mapping: dict[str, list[str]]) -> dict:
    """Top-1 accuracy and link precision/recall of the requirement -> capability matcher."""
    cap_ids = {c["name"]: c["id"] for c in load_capabilities()}
    predicted: dict[str, list[tuple[str, float]]] = {}
    for n, d in g.nodes(data=True):
        if d["type"] == "Requirement":
            predicted[d["name"]] = sorted(
                ((cap_ids[g.nodes[v]["name"]], e["score"]) for _, v, e in g.out_edges(n, data=True)
                 if e["type"] == "ADDRESSED_BY"),
                key=lambda x: -x[1],
            )
    correct_top1 = tp = fp = fn = 0
    details = []
    for req, gold in mapping.items():
        pred = [c for c, _ in predicted.get(req, [])]
        top1_ok = (pred[0] in gold) if pred else not gold
        correct_top1 += top1_ok
        tp += len(set(pred) & set(gold))
        fp += len(set(pred) - set(gold))
        fn += len(set(gold) - set(pred))
        details.append({"requirement": req, "predicted": pred, "expert": gold, "top1_ok": top1_ok})
    return {
        "top1_accuracy": correct_top1 / len(mapping),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "details": details,
    }


def _keys(extractions: list[dict]) -> tuple[set, set]:
    entities, relations = set(), set()
    for ext in extractions:
        types = {e["name"]: e["type"] for e in ext["entities"]}
        entities |= {(e["type"], normalise(e["name"])) for e in ext["entities"]}
        relations |= {
            (normalise(r["source"]), r["type"], normalise(r["target"]))
            for r in ext["relations"]
            if r["source"] in types and r["target"] in types
        }
    return entities, relations


def evaluate_extraction(candidate: list[dict], reference: list[dict]) -> dict:
    """Exact-match precision/recall after name normalisation (a strict lower bound:
    a correct entity with a different name counts as a miss)."""
    result = {}
    for label, cand, ref in zip(("entities", "relations"), _keys(candidate), _keys(reference)):
        tp = len(cand & ref)
        p = tp / len(cand) if cand else 0.0
        r = tp / len(ref) if ref else 0.0
        result[label] = {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}
    return result
