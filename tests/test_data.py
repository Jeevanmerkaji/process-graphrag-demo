"""Consistency checks on the data: the reference annotation must be valid and grounded."""

import yaml

from procgraph.evaluate import corpus_paths, load_evaluation
from procgraph.extract import DOCUMENTS_DIR, list_documents, load_reference_extractions
from procgraph.build import load_capabilities
from procgraph.metamodel import Metamodel


def test_every_document_has_a_reference_extraction():
    documents = {p.name for p in list_documents()}
    extracted = {e["document"] for e in load_reference_extractions()}
    assert documents == extracted


def test_reference_extractions_conform_to_metamodel():
    metamodel = Metamodel.load()
    for extraction in load_reference_extractions():
        assert metamodel.validate(extraction) == [], extraction["document"]


def test_evidence_quotes_are_verbatim():
    for extraction in load_reference_extractions():
        text = (DOCUMENTS_DIR / extraction["document"]).read_text(encoding="utf-8")
        for e in extraction["entities"]:
            if "evidence" in e:
                assert e["evidence"] in text, (extraction["document"], e["name"], e["evidence"])


def test_every_requirement_is_grounded_and_attached():
    for extraction in load_reference_extractions():
        attached = {r["target"] for r in extraction["relations"] if r["type"] == "HAS_REQUIREMENT"}
        for e in extraction["entities"]:
            if e["type"] == "Requirement":
                assert e.get("evidence"), e["name"]
                assert e["name"] in attached, e["name"]


def test_expected_answers_exist_in_the_corpus():
    """Keeps the comparison fair: the text baseline can find every expected fact in principle."""
    corpus = " ".join(p.read_text(encoding="utf-8") for p in corpus_paths()).lower()
    for q in load_evaluation()["questions"]:
        for fact in q["expected"]:
            assert fact.lower() in corpus, (q["question"], fact)


def test_capability_mapping_is_consistent():
    capability_ids = {c["id"] for c in load_capabilities()}
    requirement_names = {
        e["name"] for ext in load_reference_extractions() for e in ext["entities"] if e["type"] == "Requirement"
    }
    mapping = load_evaluation()["capability_mapping"]
    assert set(mapping) == requirement_names
    for caps in mapping.values():
        assert set(caps) <= capability_ids


def test_validator_catches_errors():
    metamodel = Metamodel.load()
    bad = {
        "entities": [
            {"type": "Activity", "name": "A"},
            {"type": "Role", "name": "R", "colour": "red"},
        ],
        "relations": [
            {"source": "R", "type": "PERFORMED_BY", "target": "A"},  # reversed direction
            {"source": "A", "type": "ADDRESSED_BY", "target": "R"},  # derived relation
        ],
    }
    problems = " ".join(metamodel.validate(bad))
    assert "unknown fields" in problems
    assert "expected Activity" in problems
    assert "derived relation" in problems
