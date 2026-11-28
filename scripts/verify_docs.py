#!/usr/bin/env python3
"""Check documentation anchors against the working tree.

Citations use a line range (``Sources/Foo.swift:12-40``) or a Swift symbol
(``Sources/Foo.swift#Type.member``). Line-range citations in markdown are
locked by a SHA-256 of the cited lines. Knowledge-graph nodes that name a
Swift symbol are resolved in the file and fail only when that symbol is
gone or renamed. This script does not read git history.

  python3 scripts/verify_docs.py
  python3 scripts/verify_docs.py --fix
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import doc_citations as cites

ROOT = Path(__file__).resolve().parents[1]
GRAPH = ROOT / "docs" / "knowledge-graph.json"
DOC_FILES = [
    ROOT / "AGENTS.md",
    ROOT / "docs" / "engineering_standards.md",
    ROOT / "docs" / "knowledge-graph.md",
    ROOT / "docs" / "wiki" / "01-Architecture-and-Dataflow.md",
    ROOT / "docs" / "wiki" / "02-Weights-and-Bundle.md",
    ROOT / "docs" / "wiki" / "03-Build-CI-and-Testing.md",
    ROOT / "docs" / "wiki" / "04-Memory-and-Performance.md",
]
CITATION = cites.CITATION
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


def read_lines(rel: str) -> list[str] | None:
    try:
        return lines_of(rel)
    except FileNotFoundError:
        return None


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
    """Tokens that this citation itself claims, not the next citation's subject."""
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


def load_graph() -> dict:
    return json.loads(GRAPH.read_text(encoding="utf-8"))


def save_graph(graph: dict) -> None:
    GRAPH.write_text(json.dumps(graph, indent=2) + "\n", encoding="utf-8")


def resolve_swift_symbol(rel: str, name: str) -> tuple[cites.Symbol | None, str | None]:
    lines = read_lines(rel)
    if lines is None:
        return None, f"missing file {rel}"
    if not rel.endswith(".swift") and not resolve_rel(rel).endswith(".swift"):
        return None, "not swift"
    symbols = cites.index_swift("\n".join(lines) + "\n")
    exact = [s for s in symbols if s.name == name]
    if exact:
        return exact[0], None
    return cites.find_symbol(symbols, name)


def cited_span_text(rel: str, start: int, end: int) -> str | None:
    lines = read_lines(rel)
    if lines is None or start < 1 or end < start or end > len(lines):
        return None
    return "\n".join(lines[start - 1 : end])


def check_graph_entry(
    rel: str,
    start: int,
    end: int,
    symbol: str | None,
    label: str,
    errors: list[str],
    start_only: bool = False,
) -> None:
    lines = read_lines(rel)
    if lines is None:
        errors.append(f"{label}: missing file {rel}")
        return
    if symbol:
        # File-kind graph nodes store the basename, not a Swift declaration.
        if symbol == Path(rel).name:
            return
        found, err = resolve_swift_symbol(rel, symbol)
        if found:
            return
        # Non-Swift files, or a name that is not a declaration (a product,
        # string, or token). Fall back to the cited lines still containing it.
        haystack = None
        if 1 <= start <= end <= len(lines):
            haystack = lines[start - 1] if start_only else "\n".join(lines[start - 1 : end])
        if haystack is not None and symbol in haystack:
            return
        # Last resort: the name still exists somewhere in the file.
        if symbol in "\n".join(lines):
            return
        if err and err != "not swift":
            errors.append(f"{label}: {rel}: {err}")
        else:
            errors.append(f"{label}: {rel} does not contain {symbol!r} (gone or renamed)")
        return
    if start < 1 or end < start or end > len(lines):
        errors.append(f"{label}: {rel}:{start}-{end} outside 1-{len(lines)}")


def check_markdown(errors: list[str], recorded: dict[str, str]) -> None:
    for doc in DOC_FILES:
        if not doc.is_file():
            errors.append(f"missing doc {doc.relative_to(ROOT)}")
            continue
        text = doc.read_text(encoding="utf-8")
        for line in text.splitlines():
            matches = list(CITATION.finditer(line))
            precedings = [preceding_identifier(line, match.start()) for match in matches]
            for index, match in enumerate(matches):
                cite = cites.citation_from_match(match)
                next_start = matches[index + 1].start() if index + 1 < len(matches) else len(line)
                next_preceding = precedings[index + 1] if index + 1 < len(precedings) else None
                tokens = adjacent_source_tokens(
                    line, match.start(), match.end(), next_start, next_preceding
                )
                if cite.is_symbol:
                    found, err = resolve_swift_symbol(cite.path, cite.symbol or "")
                    if err and not found:
                        # Allow a non-declaration token stored as a symbol
                        lines = read_lines(cite.path)
                        if lines is None:
                            errors.append(f"{doc.name} citation: missing file {cite.path}")
                            span = ""
                        elif cite.symbol and cite.symbol in "\n".join(lines):
                            span = "\n".join(lines)
                        else:
                            errors.append(f"{doc.name} citation: {cite.key}: {err}")
                            span = ""
                    else:
                        lines = read_lines(cite.path) or []
                        span = "\n".join(lines[found.start - 1 : found.end]) if found else ""
                    if tokens and span and not any(token in span for token in tokens):
                        errors.append(f"{doc.name}: {match.group(0)} does not contain any of {tokens}")
                    continue
                start, end = cite.line_start, cite.line_end
                try:
                    lines = lines_of(cite.path)
                except FileNotFoundError:
                    errors.append(f"{doc.name} citation: missing file {cite.path}")
                    continue
                if start < 1 or end < start or end > len(lines):
                    errors.append(
                        f"{doc.name} citation: {cite.path}:{start}-{end} outside 1-{len(lines)}"
                    )
                    continue
                span = "\n".join(lines[start - 1 : end])
                if tokens and not any(token in span for token in tokens):
                    errors.append(
                        f"{doc.name}: {match.group(0)} does not contain any of {tokens}"
                    )
                digest = cites.hash_lines(lines, start, end)
                expected = recorded.get(cite.key)
                if expected is None:
                    errors.append(f"{doc.name} citation: no content hash for {cite.key}")
                elif expected != digest:
                    errors.append(
                        f"{doc.name} citation: content hash mismatch for {cite.key} "
                        f"(content changed or moved; run python3 scripts/verify_docs.py --fix "
                        f"if the cited text moved unchanged in this file)"
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


def collect_markdown_cites() -> dict[str, cites.Citation]:
    found: dict[str, cites.Citation] = {}
    for doc in DOC_FILES:
        if not doc.is_file():
            continue
        for match in CITATION.finditer(doc.read_text(encoding="utf-8")):
            cite = cites.citation_from_match(match)
            found[cite.key] = cite
    return found


def rewrite_text(text: str, replacements: dict[str, str]) -> tuple[str, int]:
    count = 0

    def repl(match: re.Match) -> str:
        nonlocal count
        cite = cites.citation_from_match(match)
        new = replacements.get(cite.key)
        if not new or new == cite.key:
            return match.group(0)
        count += 1
        return new

    return CITATION.sub(repl, text), count


def apply_replacements(replacements: dict[str, str]) -> int:
    total = 0
    for path in DOC_FILES:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        new, n = rewrite_text(text, replacements)
        if n:
            path.write_text(new, encoding="utf-8")
            total += n
    return total


def upgrade_symbol(cite: cites.Citation) -> cites.Citation | None:
    if cite.is_symbol:
        return None
    rel = resolve_rel(cite.path)
    if not rel.endswith(".swift"):
        return None
    lines = read_lines(cite.path)
    if lines is None:
        return None
    symbols = cites.index_swift("\n".join(lines) + "\n")
    # Import the same upgrade rule used by the runtime checker.
    types = [
        s
        for s in symbols
        if s.kind in {"enum", "struct", "class", "actor", "protocol", "extension", "func", "var", "init"}
        and (s.start == cite.line_start or s.decl_line == cite.line_start)
    ]
    exact = cites.unique_symbol_for_range(symbols, cite.line_start, cite.line_end)
    if exact:
        return cites.Citation(path=cite.path, symbol=exact.name)
    candidates = []
    for symbol in types:
        if cite.line_end > symbol.end + 2:
            continue
        typelen = symbol.end - symbol.decl_line + 1
        span = cite.line_end - cite.line_start + 1
        if typelen <= 0 or span < max(1, int(0.8 * typelen)):
            continue
        candidates.append(symbol)
    if len(candidates) == 1:
        return cites.Citation(path=cite.path, symbol=candidates[0].name)
    return None


def update_graph_lines_from_symbols(graph: dict) -> int:
    changed = 0

    def walk(value) -> None:
        nonlocal changed
        if isinstance(value, dict):
            symbol = value.get("symbol")
            path = value.get("file_path")
            if symbol and path:
                found, _ = resolve_swift_symbol(path, symbol)
                if found and (
                    value.get("line_start") != found.start
                    or value.get("line_end") != found.end
                ):
                    value["line_start"] = found.start
                    value["line_end"] = found.end
                    changed += 1
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(graph)
    return changed


def fix(graph: dict) -> int:
    recorded: dict[str, str] = dict(graph.get("content_hashes") or {})
    citations = collect_markdown_cites()
    replacements: dict[str, str] = {}
    unresolved: list[str] = []
    for key, cite in sorted(citations.items()):
        if cite.is_symbol:
            continue
        lines = read_lines(cite.path)
        expected = recorded.get(key)
        if lines is None:
            unresolved.append(f"{key}: missing file")
            continue
        start, end = cite.line_start, cite.line_end
        in_range = 1 <= start <= end <= len(lines)
        current = cites.hash_lines(lines, start, end) if in_range else None
        target = cite
        if expected and (not in_range or current != expected):
            matches = cites.find_hash_windows(lines, expected, end - start + 1)
            if len(matches) == 1:
                target = cites.Citation(path=cite.path, start=matches[0][0], end=matches[0][1])
                print(f"retarget {key} -> {target.key}")
            elif len(matches) == 0:
                upgraded = upgrade_symbol(cite)
                if upgraded:
                    print(f"migrate  {key} -> {upgraded.key} (content changed; symbol still found)")
                    replacements[key] = upgraded.key
                    continue
                unresolved.append(f"{key}: cited text is gone or changed; not guessing")
                continue
            else:
                windows = ", ".join(f"{a}-{b}" for a, b in matches)
                unresolved.append(
                    f"{key}: cited text occurs {len(matches)} times ({windows}); not guessing"
                )
                continue
        upgraded = upgrade_symbol(target)
        if upgraded:
            if upgraded.key != key:
                print(f"migrate  {key} -> {upgraded.key}")
                replacements[key] = upgraded.key
            continue
        if target.key != key:
            replacements[key] = target.key

    apply_replacements(replacements)
    line_updates = update_graph_lines_from_symbols(graph)
    if line_updates:
        print(f"updated {line_updates} knowledge-graph line hint(s) from symbols")
    remaining = collect_markdown_cites()
    hashes: dict[str, str] = {}
    for key, cite in remaining.items():
        if cite.is_symbol:
            continue
        lines = read_lines(cite.path)
        if lines is None or cite.line_end > len(lines) or cite.line_start < 1:
            continue
        hashes[key] = cites.hash_lines(lines, cite.line_start, cite.line_end)
    graph["content_hashes"] = hashes
    note = graph.setdefault("metadata", {})
    note["content_hash_note"] = (
        "content_hashes are SHA-256 of markdown line-range citations, joined "
        "by newlines with a trailing newline. Swift symbol citations and "
        "knowledge-graph symbol nodes are not hashed."
    )
    save_graph(graph)
    if unresolved:
        print("unresolved citations:", file=sys.stderr)
        for item in unresolved:
            print(f"  {item}", file=sys.stderr)
        return 1
    print(
        f"fixed citations; {len(hashes)} line-range hashes, "
        f"{sum(1 for c in remaining.values() if c.is_symbol)} symbol citations"
    )
    return 0


def check_graph(graph: dict, errors: list[str]) -> None:
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
        start_only = node.get("kind") != "file"
        check_graph_entry(
            node["file_path"],
            int(node["line_start"]),
            int(node["line_end"]),
            node.get("symbol"),
            f"node {ident}",
            errors,
            start_only=start_only,
        )
    for key, entries in graph["location_index"].items():
        for entry in entries:
            check_graph_entry(
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


def main() -> int:
    if not GRAPH.is_file():
        print(f"missing {GRAPH}", file=sys.stderr)
        return 1
    graph = load_graph()
    if "--fix" in sys.argv:
        status = fix(graph)
        if status != 0:
            return status
        graph = load_graph()

    errors: list[str] = []
    check_graph(graph, errors)
    recorded = graph.get("content_hashes")
    if not isinstance(recorded, dict):
        recorded = {}
        if "--fix" not in sys.argv:
            # Hashes are required once the graph has been filled by --fix.
            # A fresh tree without hashes should run --fix first.
            errors.append("docs/knowledge-graph.json: missing content_hashes (run python3 scripts/verify_docs.py --fix)")
    check_markdown(errors, recorded if isinstance(graph.get("content_hashes"), dict) else {})
    if isinstance(graph.get("content_hashes"), dict):
        remaining = collect_markdown_cites()
        for key in graph["content_hashes"]:
            if key not in remaining:
                errors.append(f"content_hashes: stale entry {key}")
    if errors:
        print(f"{len(errors)} documentation anchor error(s):", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print(f"verified {len(graph['nodes'])} nodes against {GRAPH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
