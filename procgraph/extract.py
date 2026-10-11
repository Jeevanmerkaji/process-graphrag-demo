"""Turn process documents into metamodel-conform extractions.

Two sources are supported:

* reference extractions in data/extractions/*.yaml, annotated by hand; these
  make the demo fully reproducible without an API key, and serve as the
  ground truth for evaluating the LLM extractor;
* LLM extraction with Claude, constrained by structured outputs to the
  entity and relation types of the metamodel.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field

from .llm import call_parse
from .metamodel import DATA_DIR, Metamodel

DOCUMENTS_DIR = DATA_DIR / "documents"
EXTRACTIONS_DIR = DATA_DIR / "extractions"

EntityType = Literal["Process", "Activity", "Role", "Artifact", "Tool", "Requirement"]
RelationName = Literal[
    "HAS_ACTIVITY", "PRECEDES", "PERFORMED_BY", "USES", "PRODUCES",
    "SUPPORTED_BY", "STORED_IN", "HAS_REQUIREMENT", "TRIGGERS",
]
NeedType = Literal["information_retrieval", "consolidation", "authoring", "analysis", "monitoring", "compliance"]


class Entity(BaseModel):
    type: EntityType
    name: str = Field(description="Short canonical name, e.g. 'Design Engineer' or 'Perform impact analysis'.")
    description: str = Field(description="One sentence describing the entity.")
    evidence: Optional[str] = Field(None, description="Short verbatim quote from the document supporting the entity.")
    need_type: Optional[NeedType] = Field(None, description="Only for Requirement entities.")


class Relation(BaseModel):
    source: str = Field(description="Name of the source entity, exactly as in the entities list.")
    type: RelationName
    target: str = Field(description="Name of the target entity, exactly as in the entities list.")


class DocumentExtraction(BaseModel):
    entities: list[Entity]
    relations: list[Relation]


SYSTEM_PROMPT = """You build a process knowledge graph from engineering process documentation.
Extract entities and relations that conform exactly to this metamodel:

{metamodel}

Guidelines:
- Name activities as short verb phrases ("Perform impact analysis").
- Name roles, artifacts and tools with their usual capitalised name ("Design Engineer", "Bill of Materials", "PLM system").
- Use the same name every time an entity is mentioned. Reuse these names when the document refers to the same thing:
{known_names}
- A Requirement is a pain point, manual effort or information need stated or clearly implied in the text.
  Every Requirement must be linked to the activity it belongs to with HAS_REQUIREMENT and carry a verbatim evidence quote.
- Also extract processes that are only referenced (e.g. a process that an activity TRIGGERS).
- Extract only what the document states. Do not invent steps, roles or systems."""


def list_documents() -> list[Path]:
    return sorted(DOCUMENTS_DIR.glob("*.md"))


def load_reference_extractions() -> list[dict]:
    return [yaml.safe_load(p.read_text(encoding="utf-8")) for p in sorted(EXTRACTIONS_DIR.glob("*.yaml"))]


def extract_with_llm(doc_path: Path, metamodel: Metamodel, known_names: list[str]) -> dict:
    """Extract one document with Claude and return it in the reference format."""
    system = SYSTEM_PROMPT.format(
        metamodel=metamodel.describe(),
        known_names="\n".join(f"  - {n}" for n in sorted(known_names)) or "  (none yet)",
    )
    result: DocumentExtraction = call_parse(
        system=system,
        user=f"Document: {doc_path.name}\n\n{doc_path.read_text(encoding='utf-8')}",
        output_format=DocumentExtraction,
    )
    entities = [e.model_dump(exclude_none=True) for e in result.entities]
    relations = [r.model_dump() for r in result.relations]
    return {"document": doc_path.name, "entities": entities, "relations": relations}


def extract_all_with_llm(out_dir: Path, metamodel: Metamodel | None = None) -> list[dict]:
    """Extract every document, passing names found so far to keep entities consistent across documents."""
    metamodel = metamodel or Metamodel.load()
    out_dir.mkdir(parents=True, exist_ok=True)
    known: set[str] = set()
    extractions = []
    for doc in list_documents():
        print(f"extracting {doc.name} ...")
        extraction = extract_with_llm(doc, metamodel, sorted(known))
        for problem in metamodel.validate(extraction):
            print(f"  warning: {problem}")
        known.update(e["name"] for e in extraction["entities"] if e["type"] != "Requirement")
        (out_dir / f"{doc.stem}.yaml").write_text(
            yaml.safe_dump(extraction, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        extractions.append(extraction)
    return extractions
