"""Visualise the knowledge graph as an interactive HTML page and a static PNG."""

from __future__ import annotations

import html
import json
from pathlib import Path

import networkx as nx

COLORS = {
    "Process": "#1f4e79",
    "Activity": "#2e86c1",
    "Role": "#28b463",
    "Artifact": "#f39c12",
    "Tool": "#7f8c8d",
    "Requirement": "#c0392b",
    "AICapability": "#8e44ad",
}

HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Process Knowledge Graph</title>
<script src="https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js"></script>
<style>
  body {{ margin: 0; font-family: system-ui, sans-serif; background: #fafafa; }}
  header {{ padding: 10px 16px; border-bottom: 1px solid #ddd; background: #fff; }}
  header h1 {{ font-size: 18px; margin: 0 0 6px; }}
  .legend span {{ display: inline-block; margin-right: 12px; font-size: 13px; }}
  .legend i {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 4px; }}
  #graph {{ height: calc(100vh - 70px); }}
</style>
</head>
<body>
<header>
  <h1>Process Knowledge Graph</h1>
  <div class="legend">{legend}</div>
</header>
<div id="graph"></div>
<script>
  const nodes = new vis.DataSet({nodes});
  const edges = new vis.DataSet({edges});
  new vis.Network(document.getElementById("graph"), {{nodes, edges}}, {{
    nodes: {{shape: "dot", size: 12, font: {{size: 13}}}},
    edges: {{arrows: "to", font: {{size: 9, align: "middle"}}, color: {{color: "#bbb"}}, smooth: false}},
    physics: {{solver: "forceAtlas2Based", stabilization: {{iterations: 300}}}},
  }});
</script>
</body>
</html>
"""


def to_html(g: nx.MultiDiGraph, path: Path) -> Path:
    nodes = []
    for n, d in g.nodes(data=True):
        tooltip = f"{d['type']}: {d['name']}\n{d.get('description', '')}"
        if d.get("sources"):
            tooltip += "\nSources: " + ", ".join(d["sources"])
        nodes.append({"id": n, "label": d["name"], "title": tooltip, "color": COLORS[d["type"]],
                      "size": 20 if d["type"] == "Process" else 12})
    edges = [{"from": u, "to": v, "label": d["type"].lower()} for u, v, d in g.edges(data=True)]
    legend = "".join(f'<span><i style="background:{c}"></i>{html.escape(t)}</span>' for t, c in COLORS.items())
    path.write_text(HTML_TEMPLATE.format(nodes=json.dumps(nodes), edges=json.dumps(edges), legend=legend),
                    encoding="utf-8")
    return path


def to_png(g: nx.MultiDiGraph, path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    simple = nx.Graph(g)
    pos = nx.spring_layout(simple, k=0.45, iterations=200, seed=7)
    fig, ax = plt.subplots(figsize=(16, 11))
    nx.draw_networkx_edges(simple, pos, ax=ax, edge_color="#cccccc", width=0.8)
    for t, color in COLORS.items():
        nodelist = [n for n, d in g.nodes(data=True) if d["type"] == t]
        nx.draw_networkx_nodes(simple, pos, nodelist=nodelist, node_color=color, ax=ax,
                               node_size=260 if t == "Process" else 90)
    labels = {n: d["name"] for n, d in g.nodes(data=True) if d["type"] in ("Process", "AICapability", "Role")}
    nx.draw_networkx_labels(simple, pos, labels=labels, font_size=7, ax=ax)
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=9, label=t)
               for t, c in COLORS.items()]
    ax.legend(handles=handles, loc="upper left", frameon=False)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
