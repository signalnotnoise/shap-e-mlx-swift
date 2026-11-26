#!/usr/bin/env python3
"""Check documentation anchors against the working tree.

Line numbers in docs/knowledge-graph.json are taken from verified_commit.
This script checks the checkout that is present now. It does not require
HEAD to equal verified_commit, because a docs-only commit keeps those
source lines stable.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOC_FILES = [
    ROOT / "AGENTS.md",
    ROOT / "docs" / "engineering_standards.md",
    ROOT / "docs" / "knowledge-graph.md",
    ROOT / "docs" / "wiki" / "01-Architecture-and-Dataflow.md",
    ROOT / "docs" / "wiki" / "02-Weights-and-Bundle.md",
    ROOT / "docs" / "wiki" / "03-Build-CI-and-Testing.md",
    ROOT / "docs" / "wiki" / "04-Memory-and-Performance.md",
]
CITATION = re.compile(
    r"(?P<path>(?:Sources|Tests|scripts|docs|\.github)/[A-Za-z0-9_./-]+|Package\.swift|Package\.resolved|README\.md|run_tests\.sh):(?P<start>\d+)(?:-(?P<end>\d+))?"
)
LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def lines_of(rel: str) -> list[str]:
    path = ROOT / rel
    if not path.is_file():
        raise FileNotFoundError(rel)
    return path.read_text(encoding="utf-8").splitlines()


def check_span(
    rel: str,
    start: int,
    end: int,
    symbol: str | None,
    label: str,
    errors: list[str],
    start_only: bool = False,
) -> None:
    try:
        text = lines_of(rel)
    except FileNotFoundError:
        errors.append(f"{label}: missing file {rel}")
        return
    if start < 1 or end < start or end > len(text):
        errors.append(f"{label}: {rel}:{start}-{end} outside 1-{len(text)}")
        return
    if symbol is not None:
        haystack = text[start - 1] if start_only else "\n".join(text[start - 1 : end])
        if symbol not in haystack:
            errors.append(f"{label}: {rel}:{start}-{end} does not contain {symbol!r}")


def main() -> int:
    errors: list[str] = []
    graph_path = ROOT / "docs" / "knowledge-graph.json"
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    meta = graph["metadata"]
    for key in (
        "repo",
        "branch",
        "verified_commit",
        "verified_commit_note",
        "swift_tools_version",
        "mlx_swift_constraint",
    ):
        if not meta.get(key):
            errors.append(f"metadata.{key} is empty")
    if meta.get("swift_tools_version") != "6.2":
        errors.append("metadata.swift_tools_version is not 6.2")

    ids = set()
    for node in graph["nodes"]:
        ident = node["id"]
        if ident in ids:
            errors.append(f"duplicate node id {ident}")
        ids.add(ident)
        if node.get("kind") == "file":
            symbol = None
            start_only = False
        else:
            symbol = node.get("symbol")
            start_only = True
        check_span(
            node["file_path"],
            int(node["line_start"]),
            int(node["line_end"]),
            symbol,
            f"node {ident}",
            errors,
            start_only=start_only,
        )
    for key, entries in graph["location_index"].items():
        for entry in entries:
            check_span(
                entry["file_path"],
                int(entry["line_start"]),
                int(entry["line_end"]),
                entry.get("symbol"),
                f"location {key}",
                errors,
            )
    for edge in graph["edges"]:
        if edge["from"] not in ids or edge["to"] not in ids:
            errors.append(f"edge {edge['kind']} has a missing endpoint: {edge}")

    for doc in DOC_FILES:
        if not doc.is_file():
            errors.append(f"missing doc {doc.relative_to(ROOT)}")
            continue
        text = doc.read_text(encoding="utf-8")
        for match in CITATION.finditer(text):
            start = int(match.group("start"))
            end = int(match.group("end") or start)
            check_span(match.group("path"), start, end, None, f"{doc.name} citation", errors)
        for link in LINK.finditer(text):
            target = link.group(1)
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            target = target.split("#", 1)[0]
            if not target:
                continue
            resolved = (doc.parent / target).resolve()
            if not resolved.is_file():
                errors.append(f"{doc.relative_to(ROOT)}: broken link {link.group(1)}")

    if errors:
        print(f"{len(errors)} documentation anchor error(s):", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print(f"verified {len(graph['nodes'])} nodes against {graph_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
