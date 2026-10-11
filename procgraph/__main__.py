"""Command-line interface: python -m procgraph <command>."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from . import analysis, evaluate
from .answer import answer
from .build import build_graph, export
from .extract import extract_all_with_llm, load_reference_extractions
from .llm import MODEL, llm_available
from .metamodel import Metamodel
from .retrieve import ChunkRetriever, GraphRetriever
from .visualize import to_html, to_png


def load_extractions(directory: str | None) -> list[dict]:
    if directory is None:
        return load_reference_extractions()
    paths = sorted(Path(directory).glob("*.yaml"))
    if not paths:
        sys.exit(f"no extractions found in {directory}; run `python -m procgraph extract` first")
    return [yaml.safe_load(p.read_text(encoding="utf-8")) for p in paths]


def table(rows: list[list], header: list[str]) -> str:
    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
    fmt = lambda r: "  ".join(str(x).ljust(w) for x, w in zip(r, widths))  # noqa: E731
    return "\n".join([fmt(header), fmt(["-" * w for w in widths])] + [fmt(r) for r in rows])


def cmd_extract(args) -> None:
    if not llm_available():
        sys.exit("LLM extraction needs Claude API credentials, e.g. export ANTHROPIC_API_KEY=...")
    print(f"extracting with {MODEL}")
    extract_all_with_llm(Path(args.out))
    print(f"wrote extractions to {args.out}/; use them with --extractions {args.out}")


def cmd_build(args) -> None:
    extractions = load_extractions(args.extractions)
    metamodel = Metamodel.load()
    for ext in extractions:
        for problem in metamodel.validate(ext):
            print(f"warning ({ext['document']}): {problem}")
    g = build_graph(extractions)
    counts = {}
    for _, d in g.nodes(data=True):
        counts[d["type"]] = counts.get(d["type"], 0) + 1
    print(f"knowledge graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")
    print("  " + ", ".join(f"{t}: {c}" for t, c in counts.items()))
    out = Path(args.out)
    paths = export(g, out) + [to_html(g, out / "graph.html")]
    if not args.no_png:
        paths.append(to_png(g, out / "graph.png"))
    for p in paths:
        print(f"wrote {p}")


def cmd_ask(args) -> None:
    g = build_graph(load_extractions(args.extractions))
    if args.baseline:
        retrieval = ChunkRetriever(evaluate.corpus_paths()).retrieve(args.question)
        print("Retrieved chunks: " + ", ".join(retrieval.items))
    else:
        retrieval = GraphRetriever(g).retrieve(args.question)
        print("Seed entities: " + ", ".join(g.nodes[n]["name"] for n in retrieval.seeds))
    result = None if args.no_llm else answer(retrieval)
    if result is None or args.show_context:
        print("\n--- retrieved context ---")
        print(retrieval.context)
    if result is not None:
        print("\n--- answer ---")
        print(result)
    elif not args.no_llm:
        print("\n(no Claude API credentials found, showing retrieved context only; "
              "set ANTHROPIC_API_KEY to generate an answer)")


def cmd_gaps(args) -> None:
    g = build_graph(load_extractions(args.extractions))
    print("DOCUMENTATION GAPS\n")
    for title, items in analysis.documentation_gaps(g).items():
        print(f"{title} ({len(items)}):")
        for item in items:
            print(f"  - {item}")
        print()
    print("CROSS-PROCESS ARTIFACTS (produced in one process, used in another)\n")
    rows = [[a, ", ".join(p) or "-", ", ".join(c) or "-"] for a, p, c in analysis.cross_process_links(g)]
    print(table(rows, ["Artifact", "Produced in", "Used in"]))
    print("\nAI OPPORTUNITIES (priority = effort x match score x maturity weight)\n")
    rows = [[r["requirement"], r["process"], r["effort_hours_per_month"], r["capability"], r["maturity"],
             r["score"], r["priority"]] for r in analysis.ai_opportunities(g)]
    print(table(rows, ["Requirement", "Process", "h/month", "Best AI capability", "Maturity", "Match",
                       "Priority"]))


def cmd_evaluate(args) -> None:
    g = build_graph(load_extractions(args.extractions))
    data = evaluate.load_evaluation()

    print("1. RETRIEVAL: context recall of expected facts (same context size for both)\n")
    rows = evaluate.evaluate_retrieval(g, data["questions"])
    print(table([[r["hops"], f"{r['graph']:.2f}", f"{r['chunks']:.2f}", r["question"]] for r in rows],
                ["Hops", "GraphRAG", "Chunk RAG", "Question"]))
    print()
    for hops in ("single", "multi", None):
        n = sum(1 for r in rows if hops is None or r["hops"] == hops)
        print(f"mean recall, {hops or 'all'} ({n} questions): GraphRAG {evaluate.mean(rows, 'graph', hops):.2f}, "
              f"chunk RAG {evaluate.mean(rows, 'chunks', hops):.2f}")
    print(f"average context size {evaluate.mean(rows, 'chars'):.0f} characters "
          f"(whole corpus: {evaluate.corpus_chars()} characters)")
    if args.sweep:
        print("\nAblation: schema-aware boost (1 = off) and number of graph nodes in the context\n")
        print(table([[r["type_boost"], r["top_n"], r["chars"], f"{r['graph_multi']:.2f}", f"{r['chunks_multi']:.2f}",
                      f"{r['graph_all']:.2f}", f"{r['chunks_all']:.2f}"] for r in evaluate.sweep(g, data["questions"])],
                    ["Boost", "Nodes", "Chars", "Graph multi", "Chunk multi", "Graph all", "Chunk all"]))
    if args.verbose:
        for r in rows:
            print(f"\n{r['question']}\n  missed by GraphRAG: {r['missing_graph']}\n"
                  f"  missed by chunk RAG: {r['missing_chunks']}")

    print("\n2. CAPABILITY MATCHING vs. expert mapping\n")
    m = evaluate.evaluate_matching(g, data["capability_mapping"])
    print(f"top-1 accuracy {m['top1_accuracy']:.2f}, link precision {m['precision']:.2f}, "
          f"link recall {m['recall']:.2f}")
    wrong = [d for d in m["details"] if not d["top1_ok"]]
    if wrong:
        print("disagreements with the expert:")
        for d in wrong:
            print(f"  - {d['requirement']}: matcher {d['predicted'] or '(none)'}, expert {d['expert'] or '(none)'}")

    if args.extractions:
        print("\n3. EXTRACTION vs. reference annotation\n")
        res = evaluate.evaluate_extraction(load_extractions(args.extractions), load_reference_extractions())
        for label, s in res.items():
            print(f"{label}: precision {s['precision']:.2f}, recall {s['recall']:.2f}, F1 {s['f1']:.2f}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m procgraph", description=__doc__)
    parser.add_argument("--extractions", help="folder with extractions to use instead of the reference annotation")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("extract", help="extract the documents with Claude (needs an API key)")
    p.add_argument("--out", default="build/llm_extractions")
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser("build", help="build the graph and export JSON, GraphML, HTML and PNG")
    p.add_argument("--out", default="build")
    p.add_argument("--no-png", action="store_true")
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("ask", help="answer a question with GraphRAG")
    p.add_argument("question")
    p.add_argument("--baseline", action="store_true", help="use plain text-chunk RAG instead of the graph")
    p.add_argument("--show-context", action="store_true", help="also print the retrieved context")
    p.add_argument("--no-llm", action="store_true", help="only retrieve, do not call Claude")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("gaps", help="documentation gaps and ranked AI opportunities")
    p.set_defaults(func=cmd_gaps)

    p = sub.add_parser("evaluate", help="evaluate retrieval, capability matching and (optionally) extraction")
    p.add_argument("-v", "--verbose", action="store_true", help="list the missed facts per question")
    p.add_argument("--sweep", action="store_true", help="also run the retrieval ablation")
    p.set_defaults(func=cmd_evaluate)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
