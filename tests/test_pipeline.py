"""End-to-end behaviour of graph building, retrieval and analysis (no LLM needed)."""

import pytest

from procgraph import analysis
from procgraph.build import build_graph, node_id
from procgraph.extract import load_reference_extractions
from procgraph.metamodel import Metamodel
from procgraph.retrieve import ChunkRetriever, GraphRetriever, chunk_documents
from procgraph.evaluate import corpus_paths


@pytest.fixture(scope="module")
def graph():
    return build_graph(load_reference_extractions())


def test_shared_entities_are_merged_across_documents(graph):
    test_report = graph.nodes[node_id("Artifact", "Test Report")]
    assert len(test_report["sources"]) >= 3


def test_failed_test_triggers_change_management(graph):
    assert graph.has_edge(node_id("Activity", "Report test failure"),
                          node_id("Process", "Engineering Change Management"), key="TRIGGERS")


@pytest.mark.parametrize("requirement, capability", [
    ("Detect anomalies in measurement data", "Anomaly detection in measurement data"),
    ("Identify affected parts", "Dependency and impact analysis"),
    ("Reduce meshing effort", "Automated geometry and mesh preparation"),
    ("Find earlier similar changes", "Semantic search and retrieval"),
])
def test_capability_matching(graph, requirement, capability):
    assert graph.has_edge(node_id("Requirement", requirement), node_id("AICapability", capability),
                          key="ADDRESSED_BY")


def test_requested_types():
    metamodel = Metamodel.load()
    assert "Role" in metamodel.requested_types("Who releases the Design FMEA?")
    assert "AICapability" in metamodel.requested_types("Which AI capabilities could help?")
    assert "Tool" not in metamodel.requested_types("Who releases the Design FMEA?")


def test_graph_retrieval_crosses_documents(graph):
    retrieval = GraphRetriever(graph).retrieve("Which roles work with the bill of materials?")
    for role in ("Change Coordinator", "FMEA Moderator", "Prototype Workshop"):
        assert role in retrieval.context


def test_chunking_splits_table_rows():
    chunks = chunk_documents(corpus_paths())
    rows = [text for _, text in chunks if "| Run simulation |" in text]
    assert len(rows) == 1 and "| Step |" in rows[0]  # each row keeps the table header


def test_chunk_retriever_respects_budget():
    retriever = ChunkRetriever(corpus_paths())
    retrieval = retriever.retrieve("simulation report", char_budget=800)
    texts = dict(retriever.chunks)
    assert retrieval.items
    assert sum(len(texts[c]) for c in retrieval.items) <= 800 or len(retrieval.items) == 1


def test_gap_analysis(graph):
    gaps = analysis.documentation_gaps(graph)
    used_not_produced = next(v for k, v in gaps.items() if k.startswith("Artifacts used but produced"))
    assert "Test Request" in used_not_produced
    assert "Link test and simulation data" in gaps["Requirements no AI capability addresses"]


def test_opportunities_are_ranked(graph):
    rows = analysis.ai_opportunities(graph)
    priorities = [r["priority"] for r in rows]
    assert priorities == sorted(priorities, reverse=True)
    assert rows[0]["capability"] != "(none)"
