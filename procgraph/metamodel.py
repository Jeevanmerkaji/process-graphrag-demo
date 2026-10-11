"""Load the metamodel and validate extractions against it."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ENTITY_FIELDS = {"type", "name", "description", "evidence", "need_type", "effort_hours_per_month"}


@dataclass(frozen=True)
class RelationType:
    name: str
    domain: str
    range: str
    description: str
    derived: bool = False


@dataclass(frozen=True)
class Metamodel:
    entity_types: dict[str, str]  # name -> description
    relation_types: dict[str, RelationType]
    query_terms: dict[str, list[str]]  # entity type -> words that signal a question asks for it

    @classmethod
    def load(cls, path: Path = DATA_DIR / "metamodel.yaml") -> "Metamodel":
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        entity_types = {name: spec["description"].strip() for name, spec in raw["entity_types"].items()}
        relation_types = {
            name: RelationType(
                name=name,
                domain=spec["domain"],
                range=spec["range"],
                description=spec["description"].strip(),
                derived=spec.get("derived", False),
            )
            for name, spec in raw["relation_types"].items()
        }
        query_terms = {name: spec.get("query_terms", []) for name, spec in raw["entity_types"].items()}
        return cls(entity_types, relation_types, query_terms)

    @property
    def extractable_relations(self) -> list[str]:
        return [name for name, r in self.relation_types.items() if not r.derived]

    def requested_types(self, question: str) -> set[str]:
        """Entity types the question asks for, e.g. "Which roles ..." -> {"Role"}."""
        text = question.lower()
        return {
            t for t, terms in self.query_terms.items()
            if any(re.search(rf"\b{re.escape(term)}\b", text) for term in terms)
        }

    def validate(self, extraction: dict) -> list[str]:
        """Return a list of human-readable problems; an empty list means the extraction is valid."""
        problems = []
        types_by_name: dict[str, str] = {}
        for e in extraction.get("entities", []):
            unknown = set(e) - ENTITY_FIELDS
            if unknown:
                problems.append(f"unknown fields {sorted(unknown)} on {e.get('name')!r}")
            if e["type"] not in self.entity_types:
                problems.append(f"unknown entity type {e['type']!r} for {e['name']!r}")
            if e["name"] in types_by_name and types_by_name[e["name"]] != e["type"]:
                problems.append(f"{e['name']!r} declared with two types")
            types_by_name[e["name"]] = e["type"]

        for r in extraction.get("relations", []):
            label = f"{r['source']} -{r['type']}-> {r['target']}"
            rel = self.relation_types.get(r["type"])
            if rel is None:
                problems.append(f"unknown relation type in {label}")
                continue
            if rel.derived:
                problems.append(f"derived relation must not be extracted: {label}")
            for end, expected in (("source", rel.domain), ("target", rel.range)):
                actual = types_by_name.get(r[end])
                if actual is None:
                    problems.append(f"{end} {r[end]!r} is not a declared entity in {label}")
                elif actual != expected:
                    problems.append(f"{end} of {label} is a {actual}, expected {expected}")
        return problems

    def describe(self) -> str:
        """Plain-text description of the metamodel, used in the extraction prompt."""
        lines = ["Entity types:"]
        lines += [f"- {name}: {desc}" for name, desc in self.entity_types.items() if name != "AICapability"]
        lines.append("Relation types (source type -> target type):")
        lines += [
            f"- {r.name} ({r.domain} -> {r.range}): {r.description}"
            for r in self.relation_types.values()
            if not r.derived
        ]
        return "\n".join(lines)
