"""Retrieval for question answering.

GraphRetriever (GraphRAG, "local search"):
  1. entity linking: find the graph nodes that best match the question;
  2. expansion: run personalised PageRank from those seed nodes, so that
     nodes that are well connected to the seeds (also across documents) rank high;
  3. schema-aware ranking: boost nodes whose type the question asks for, using the
     query terms in the metamodel ("Which roles ..." favours Role nodes);
  4. context: serialise the top nodes and the relations between them.

ChunkRetriever (baseline, classic RAG):
  rank text chunks of the source documents by similarity to the question.

Both use the same TF-IDF text similarity, so the comparison isolates the
effect of the graph structure rather than of a better embedding model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .metamodel import Metamodel


def _vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(4, 5), sublinear_tf=True)


@dataclass
class Retrieval:
    question: str
    context: str
    items: list = field(default_factory=list)  # node ids (graph) or chunk ids (baseline)
    seeds: list = field(default_factory=list)


def node_text(d: dict) -> str:
    parts = [d["type"], d["name"] + ".", d.get("description", "")]
    if d.get("process"):
        parts.append(f"Part of {d['process']}.")
    parts += [ev["quote"] for ev in d.get("evidence", [])]
    return " ".join(parts)


class GraphRetriever:
    def __init__(self, graph: nx.MultiDiGraph, metamodel: Metamodel | None = None, num_seeds: int = 4,
                 top_n: int = 20, alpha: float = 0.7, type_boost: float = 3.0):
        self.g = graph
        self.metamodel = metamodel or Metamodel.load()
        self.num_seeds = num_seeds
        self.top_n = top_n
        self.alpha = alpha  # PageRank continuation probability; lower = stay closer to the seeds
        self.type_boost = type_boost  # 1.0 switches schema-aware ranking off
        self.node_ids = list(graph.nodes)
        self.vectorizer = _vectorizer()
        self.matrix = self.vectorizer.fit_transform([node_text(graph.nodes[n]) for n in self.node_ids])
        # Undirected view for the walk: "who uses my output" matters as much as "what I use".
        self.walk_graph = nx.Graph(graph)

    def link_entities(self, question: str) -> dict[str, float]:
        scores = cosine_similarity(self.vectorizer.transform([question]), self.matrix)[0]
        ranked = scores.argsort()[::-1][: self.num_seeds]
        return {self.node_ids[i]: float(scores[i]) for i in ranked if scores[i] > 0}

    def retrieve(self, question: str) -> Retrieval:
        seeds = self.link_entities(question)
        if not seeds:
            return Retrieval(question, "No matching entities found in the knowledge graph.")
        ppr = nx.pagerank(self.walk_graph, alpha=self.alpha, personalization=seeds)
        wanted = self.metamodel.requested_types(question)
        score = {n: p * (self.type_boost if self.g.nodes[n]["type"] in wanted else 1.0) for n, p in ppr.items()}
        top = sorted(score, key=score.get, reverse=True)[: self.top_n]
        return Retrieval(question, self.serialise(top), items=top, seeds=list(seeds))

    def serialise(self, node_ids: list[str]) -> str:
        selected = set(node_ids)
        lines = ["Entities:"]
        for n in node_ids:
            d = self.g.nodes[n]
            line = f"- [{d['type']}] {d['name']}"
            if d.get("process"):
                line += f" (in {d['process']})"
            line += f": {d.get('description', '')}"
            if d.get("effort_hours_per_month"):
                line += f" Estimated effort: {d['effort_hours_per_month']} h/month."
            for ev in d.get("evidence", []):
                line += f' Source {ev["document"]}: "{ev["quote"]}"'
            lines.append(line)
        lines.append("Relations:")
        for u, v, data in self.g.edges(data=True):
            if u in selected and v in selected:
                rel = f"- {self.g.nodes[u]['name']} --{data['type']}--> {self.g.nodes[v]['name']}"
                if "score" in data:
                    rel += f" (match score {data['score']})"
                lines.append(rel)
        return "\n".join(lines)


def chunk_documents(paths: list[Path]) -> list[tuple[str, str]]:
    """Split documents into paragraphs; every table row is its own chunk. Returns (chunk id, text)."""
    chunks = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        blocks = re.split(r"\n\s*\n", text)
        for block in blocks:
            lines = block.strip().splitlines()
            if not lines:
                continue
            if all(line.startswith("|") for line in lines):
                header = lines[0]
                pieces = [f"{header}\n{row}" for row in lines[2:]]  # skip the |---| separator
            else:
                pieces = ["\n".join(lines)]
            for piece in pieces:
                chunks.append((f"{path.name}#{len(chunks)}", f"[{path.name}] {piece}"))
    return chunks


class ChunkRetriever:
    def __init__(self, paths: list[Path]):
        self.chunks = chunk_documents(paths)
        self.vectorizer = _vectorizer()
        self.matrix = self.vectorizer.fit_transform([text for _, text in self.chunks])

    def retrieve(self, question: str, char_budget: int = 2500) -> Retrieval:
        """Take the best-matching chunks until the character budget is used up."""
        scores = cosine_similarity(self.vectorizer.transform([question]), self.matrix)[0]
        picked, used = [], 0
        for i in scores.argsort()[::-1]:
            cid, text = self.chunks[i]
            if picked and used + len(text) > char_budget:
                break
            picked.append(i)
            used += len(text)
        context = "\n\n".join(self.chunks[i][1] for i in picked)
        return Retrieval(question, context, items=[self.chunks[i][0] for i in picked])
