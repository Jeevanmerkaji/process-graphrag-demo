"""Structural analyses of the process knowledge graph.

These need no LLM: they are graph queries over the metamodel, the kind of
analysis that becomes possible once documentation is machine-readable.
"""

from __future__ import annotations

import networkx as nx

MATURITY_WEIGHT = {"high": 1.0, "medium": 0.7, "low": 0.4}


def _nodes(g: nx.MultiDiGraph, entity_type: str) -> list[str]:
    return [n for n, d in g.nodes(data=True) if d["type"] == entity_type]


def _out(g: nx.MultiDiGraph, n: str, rel: str) -> list[str]:
    return [v for _, v, d in g.out_edges(n, data=True) if d["type"] == rel]


def _in(g: nx.MultiDiGraph, n: str, rel: str) -> list[str]:
    return [u for u, _, d in g.in_edges(n, data=True) if d["type"] == rel]


def documentation_gaps(g: nx.MultiDiGraph) -> dict[str, list[str]]:
    """Places where the documented process landscape is incomplete or disconnected."""
    name = lambda n: g.nodes[n]["name"]  # noqa: E731
    activities = _nodes(g, "Activity")
    artifacts = _nodes(g, "Artifact")
    return {
        "Activities without a responsible role": [name(a) for a in activities if not _out(g, a, "PERFORMED_BY")],
        "Activities without documented tool support": [name(a) for a in activities if not _out(g, a, "SUPPORTED_BY")],
        "Artifacts used but produced by no documented activity (interface to undocumented processes)": [
            name(a) for a in artifacts if _in(g, a, "USES") and not _in(g, a, "PRODUCES")
        ],
        "Artifacts produced but used by no documented activity (possible waste or missing consumer)": [
            name(a) for a in artifacts if _in(g, a, "PRODUCES") and not _in(g, a, "USES")
        ],
        "Artifacts without a documented storage system": [name(a) for a in artifacts if not _out(g, a, "STORED_IN")],
        "Requirements no AI capability addresses": [
            name(r) for r in _nodes(g, "Requirement") if not _out(g, r, "ADDRESSED_BY")
        ],
    }


def ai_opportunities(g: nx.MultiDiGraph) -> list[dict]:
    """Rank requirements by potential benefit of AI support.

    priority = effort (h/month) x match score x capability maturity weight.
    A transparent first-order estimate for discussion with process owners,
    not a business case.
    """
    rows = []
    for r in _nodes(g, "Requirement"):
        d = g.nodes[r]
        activity = _in(g, r, "HAS_REQUIREMENT")
        best = None
        for _, cap, e in g.out_edges(r, data=True):
            if e["type"] != "ADDRESSED_BY":
                continue
            cd = g.nodes[cap]
            priority = d.get("effort_hours_per_month", 0) * e["score"] * MATURITY_WEIGHT[cd["maturity"]]
            if best is None or priority > best["priority"]:
                best = {"capability": cd["name"], "maturity": cd["maturity"], "score": e["score"],
                        "priority": round(priority, 1)}
        rows.append({
            "requirement": d["name"],
            "activity": g.nodes[activity[0]]["name"] if activity else "",
            "process": g.nodes[activity[0]].get("process", "") if activity else "",
            "effort_hours_per_month": d.get("effort_hours_per_month", 0),
            **(best or {"capability": "(none)", "maturity": "", "score": 0.0, "priority": 0.0}),
        })
    return sorted(rows, key=lambda row: row["priority"], reverse=True)


def cross_process_links(g: nx.MultiDiGraph) -> list[tuple[str, list[str], list[str]]]:
    """Artifacts that connect processes: (artifact, producing processes, consuming processes)."""
    links = []
    for a in _nodes(g, "Artifact"):
        producers = sorted({g.nodes[x].get("process", "?") for x in _in(g, a, "PRODUCES")})
        consumers = sorted({g.nodes[x].get("process", "?") for x in _in(g, a, "USES")})
        if len(set(producers) | set(consumers)) > 1:
            links.append((g.nodes[a]["name"], producers, consumers))
    return sorted(links)
