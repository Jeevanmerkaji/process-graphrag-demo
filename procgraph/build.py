"""Build the process knowledge graph.

1. Merge the per-document extractions into one graph. Entities with the same
   type and (normalised) name become one node, which is what connects the
   processes to each other, e.g. the "Test Report" produced in prototype
   testing is the same node as the one used in the quality gate review.
2. Add the AI capability taxonomy as nodes.
3. Link each process requirement to the AI capabilities that could address it
   (ADDRESSED_BY edges), using TF-IDF similarity between their descriptions.
   This is a transparent, deterministic baseline; `evaluate` measures it
   against an expert mapping.
"""

from __future__ import annotations

import re

import networkx as nx
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .metamodel import DATA_DIR

MATCH_THRESHOLD = 0.15  # minimum similarity for a requirement -> capability link
MATCH_RELATIVE = 0.9  # also keep runner-up capabilities scoring at least this fraction of the best


def normalise(name: str) -> str:
    return re.sub(r"\s+", " ", name).strip().lower()


def node_id(entity_type: str, name: str) -> str:
    return f"{entity_type}:{normalise(name)}"


def load_capabilities() -> list[dict]:
    return yaml.safe_load((DATA_DIR / "ai_capabilities.yaml").read_text(encoding="utf-8"))["capabilities"]


def build_graph(extractions: list[dict], capabilities: list[dict] | None = None) -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    for ext in extractions:
        doc = ext["document"]
        names = {}  # entity name -> node id, within this document
        for e in ext["entities"]:
            nid = node_id(e["type"], e["name"])
            names[e["name"]] = nid
            if nid not in g:
                g.add_node(nid, type=e["type"], name=e["name"], description=e.get("description", ""),
                           sources=[], evidence=[])
            node = g.nodes[nid]
            if doc not in node["sources"]:
                node["sources"].append(doc)
            if e.get("evidence"):
                node["evidence"].append({"document": doc, "quote": e["evidence"]})
            for attr in ("need_type", "effort_hours_per_month"):
                if e.get(attr) is not None:
                    node[attr] = e[attr]
        for r in ext["relations"]:
            u, v = names[r["source"]], names[r["target"]]
            if not g.has_edge(u, v, key=r["type"]):
                g.add_edge(u, v, key=r["type"], type=r["type"], sources=[doc])
            elif doc not in g.edges[u, v, r["type"]]["sources"]:
                g.edges[u, v, r["type"]]["sources"].append(doc)

    # Every activity belongs to its process; store it on the node for display.
    for u, v, data in g.edges(data=True):
        if data["type"] == "HAS_ACTIVITY":
            g.nodes[v]["process"] = g.nodes[u]["name"]

    capabilities = load_capabilities() if capabilities is None else capabilities
    for cap in capabilities:
        g.add_node(node_id("AICapability", cap["name"]), type="AICapability", name=cap["name"],
                   description=cap["description"].strip(), category=cap["category"],
                   maturity=cap["maturity"], capability_id=cap["id"], sources=["ai_capabilities.yaml"],
                   evidence=[])
    link_requirements_to_capabilities(g, capabilities)
    return g


def match_capabilities(requirements: dict[str, str], capabilities: list[dict]) -> list[tuple[str, str, float]]:
    """Score every requirement against every capability.

    requirements: requirement id -> text. Returns (requirement id, capability id, score)
    for the links that pass the thresholds.
    """
    cap_texts = [
        f"{c['name']}. {c['description']} {' '.join(c.get('keywords', []))}" for c in capabilities
    ]
    req_ids = list(requirements)
    # Character n-grams within words are robust to word forms ("consolidated" vs "consolidation").
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(4, 5), sublinear_tf=True)
    vectorizer.fit(cap_texts + [requirements[r] for r in req_ids])
    sims = cosine_similarity(vectorizer.transform([requirements[r] for r in req_ids]),
                             vectorizer.transform(cap_texts))
    links = []
    for i, rid in enumerate(req_ids):
        best = sims[i].max()
        for j, cap in enumerate(capabilities):
            score = float(sims[i, j])
            if score >= MATCH_THRESHOLD and score >= MATCH_RELATIVE * best:
                links.append((rid, cap["id"], round(score, 3)))
    return links


def link_requirements_to_capabilities(g: nx.MultiDiGraph, capabilities: list[dict]) -> None:
    requirements = {
        nid: f"{d['name']}. {d['description']} " + " ".join(ev["quote"] for ev in d["evidence"])
        for nid, d in g.nodes(data=True)
        if d["type"] == "Requirement"
    }
    if not requirements:
        return
    cap_nodes = {c["id"]: node_id("AICapability", c["name"]) for c in capabilities}
    for rid, cap_id, score in match_capabilities(requirements, capabilities):
        g.add_edge(rid, cap_nodes[cap_id], key="ADDRESSED_BY", type="ADDRESSED_BY", score=score,
                   sources=["capability matcher"])


def export(g: nx.MultiDiGraph, out_dir) -> list:
    """Write the graph as JSON (node-link) and GraphML (for Gephi, yEd, Neo4j import)."""
    import json
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "graph.json"
    json_path.write_text(json.dumps(nx.node_link_data(g, edges="edges"), indent=2), encoding="utf-8")

    # GraphML only supports scalar attributes.
    flat = nx.MultiDiGraph()
    for nid, d in g.nodes(data=True):
        flat.add_node(nid, **{k: (", ".join(map(str, v)) if isinstance(v, list) and k != "evidence"
                                  else " | ".join(e["quote"] for e in v) if k == "evidence" else v)
                              for k, v in d.items()})
    for u, v, k, d in g.edges(keys=True, data=True):
        flat.add_edge(u, v, key=k, **{a: (", ".join(b) if isinstance(b, list) else b) for a, b in d.items()})
    graphml_path = out / "graph.graphml"
    nx.write_graphml(flat, graphml_path)
    return [json_path, graphml_path]
