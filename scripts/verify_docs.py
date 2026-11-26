#!/usr/bin/env python3
"""Check documentation anchors against the working tree.

Citations are checked against the files in this checkout. This script does
not read git history, so a squash or rebase that drops verified_commit is
not a failure. A cited range must exist, and when the same sentence names a
source token after the citation, that range must contain the token.
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
    r"(?P<path>(?:Sources|Tests|scripts|docs|\.github)/[A-Za-z0-9_./-]+|[A-Za-z0-9_.-]+\.(?:swift|py|yml|md|sh|metal)|Package\.swift|Package\.resolved|README\.md|run_tests\.sh):(?P<start>\d+)(?:-(?P<end>\d+))?"
)
LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
BACKTICK = re.compile(r"`([^`\n]+)`")


def resolve_rel(rel: str) -> str:
    direct = ROOT / rel
    if direct.is_file():
        return rel
    if "/" in rel:
        return rel
    found = [
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob(rel)
        if path.is_file() and ".git" not in path.parts
    ]
    if len(found) == 1:
        return found[0]
    return rel


def lines_of(rel: str) -> list[str]:
    path = ROOT / resolve_rel(rel)
    if not path.is_file():
        raise FileNotFoundError(rel)
    return path.read_text(encoding="utf-8").splitlines()


def is_literal_source_token(token: str) -> bool:
    """Repo ids and full revisions are copied from source, not paraphrased."""

    if "<" in token or ">" in token:
        return False
    if re.fullmatch(r"[0-9a-f]{40}", token):
        return True
    if token.startswith(".") or token.startswith("<"):
        return False
    parts = token.split("/")
    if len(parts) != 2 or not parts[0][:1].isalpha() or not parts[1]:
        return False
    return re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", token) is not None


def is_citation_token(token: str) -> bool:
    return CITATION.search(token) is not None


def preceding_identifier(line: str, citation_start: int) -> str | None:
    prefix = line[:citation_start].rstrip()
    if prefix.endswith("`"):
        prefix = prefix[:-1].rstrip()
    if prefix.endswith("("):
        prefix = prefix[:-1].rstrip()
    before = re.search(r"`([^`\s]+)`$", prefix)
    if before and not is_citation_token(before.group(1)):
        return before.group(1)
    return None


def adjacent_source_tokens(
    line: str,
    citation_start: int,
    citation_end: int,
    next_citation_start: int,
    next_citation_preceding: str | None,
) -> list[str]:
    """Tokens that this citation itself claims, not the next citation's subject.

    `file:lines` followed by `signalNotNoise/shap-e-mlx` must contain that id.
    `symbol` (`file:lines`) must contain symbol. A later `other` (`file:other`)
    does not belong to this citation.
    """

    tokens: list[str] = []
    before = preceding_identifier(line, citation_start)
    if before:
        tokens.append(before)

    window = line[citation_end:next_citation_start]
    period = window.find(".")
    if period != -1:
        window = window[:period]
    if window.startswith("`"):
        window = window[1:]
    for token in BACKTICK.findall(window):
        if any(character.isspace() for character in token) or is_citation_token(token):
            continue
        if token == next_citation_preceding:
            continue
        tokens.append(token)
        break
    return [token for token in tokens if is_literal_source_token(token)]


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
        for line in text.splitlines():
            matches = list(CITATION.finditer(line))
            precedings = [preceding_identifier(line, match.start()) for match in matches]
            for index, match in enumerate(matches):
                start = int(match.group("start"))
                end = int(match.group("end") or start)
                check_span(match.group("path"), start, end, None, f"{doc.name} citation", errors)
                next_start = matches[index + 1].start() if index + 1 < len(matches) else len(line)
                next_preceding = precedings[index + 1] if index + 1 < len(precedings) else None
                span = ""
                cited = ROOT / resolve_rel(match.group("path"))
                if cited.is_file():
                    span = "\n".join(lines_of(match.group("path"))[start - 1 : end])
                tokens = adjacent_source_tokens(
                    line, match.start(), match.end(), next_start, next_preceding
                )
                if tokens and not any(token in span for token in tokens):
                    errors.append(
                        f"{doc.name}: {match.group(0)} does not contain any of {tokens}"
                    )
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
