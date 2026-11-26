#!/usr/bin/env python3
"""Download the pinned Shap-E MLX runtime bundle and verify checksums.

The snapshot is https://huggingface.co/signalNotNoise/shap-e-mlx at the
revision recorded below. Pass the downloaded directory to ShapERuntime as
modelBundleURL, and pass bpe_simple_vocab_16e6.txt.gz (a file at the bundle
root, not a tensor family) as bpeVocabularyURL.

Checksum lines match the bundle's checksums.sha256: 64 hex characters, two
spaces, then a relative path. checksums.sha256 itself is not an entry.

`--text-only` still requires every checksum entry in the text scope
(root files, `licenses/*`, and the five text-family `*.bin` files).
Missing entries outside that scope are skipped. A checksum list that
omits a tensor path declared by manifest.json fails even when every
listed file matches. The Swift unit tests use the checked-in stub and
do not download this bundle.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import sys
from pathlib import Path


DEFAULT_REPO_ID = "signalNotNoise/shap-e-mlx"
DEFAULT_REVISION = "d71f70a09b9c03535188d006a719edda5c19a6eb"

TEXT_FAMILIES = (
    "clip_text",
    "text300m",
    "diffusion",
    "decoder_params",
    "renderer",
)
IMAGE_FAMILIES = (
    "clip_vision",
    "image300m",
)
ROOT_FILES = (
    "manifest.json",
    "bpe_simple_vocab_16e6.txt.gz",
    "checksums.sha256",
    "provenance.json",
    "LICENSE",
    "README.md",
)


def patterns_for_scope(*, text_only: bool) -> list[str] | None:
    if not text_only:
        return None
    patterns = list(ROOT_FILES)
    patterns.append("licenses/*")
    patterns.extend(f"{family}/*.bin" for family in TEXT_FAMILIES)
    return patterns


def parse_checksum_lines(lines: list[str]) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for line_number, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line:
            continue
        if "  " not in line:
            raise ValueError(f"Malformed checksums.sha256 line {line_number}: {raw!r}")
        digest, name = line.split("  ", 1)
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest.lower()) or not name:
            raise ValueError(f"Malformed checksums.sha256 line {line_number}: {raw!r}")
        entries.append((digest.lower(), name))
    return entries


def is_text_scope_path(relative_name: str) -> bool:
    """True when a checksum path is part of a `--text-only` download."""

    normalized = relative_name.replace("\\", "/").lstrip("./")
    patterns = patterns_for_scope(text_only=True)
    assert patterns is not None
    return any(fnmatch.fnmatch(normalized, pattern) for pattern in patterns)


def ensure_text_only_output_is_clean(output: Path) -> None:
    present = [
        family
        for family in IMAGE_FAMILIES
        if (output / family).is_dir() and any((output / family).iterdir())
    ]
    if present:
        joined = ", ".join(present)
        raise ValueError(
            f"--text-only output already contains image families ({joined}): {output}. "
            "Choose an empty output directory."
        )


def manifest_tensor_paths(manifest_path: Path) -> list[str]:
    """Return tensor blob paths declared by a format-v1 runtime manifest."""

    import json

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"manifest.json is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("manifest.json must be a JSON object")
    families = payload.get("families")
    if not isinstance(families, dict):
        raise ValueError("manifest.json is missing an object field 'families'")

    paths: list[str] = []
    for family_name, family in families.items():
        if not isinstance(family, dict):
            raise ValueError(f"manifest family {family_name!r} is not an object")
        tensors = family.get("tensors", {})
        if not isinstance(tensors, dict):
            raise ValueError(f"manifest family {family_name!r} tensors is not an object")
        for tensor_name, entry in tensors.items():
            path = entry.get("path") if isinstance(entry, dict) else None
            if not isinstance(path, str) or not path:
                raise ValueError(
                    f"manifest tensor {family_name}/{tensor_name} is missing a path"
                )
            paths.append(path.replace("\\", "/").lstrip("./"))
    return paths


def ensure_checksum_list_covers_manifest(bundle: Path, checksum_names: set[str]) -> None:
    """Fail when checksums.sha256 omits a tensor path the manifest declares.

    A truncated list can still match every file it bothers to name. The
    manifest is the list of blobs the runtime will open, so each of those
    paths has to appear in the checksum list. This applies to text-only
    verification as well: image paths may be absent from disk, but they
    still have to be named when the manifest includes them.
    """

    manifest_path = bundle / "manifest.json"
    if not manifest_path.is_file():
        return
    missing = [
        path for path in manifest_tensor_paths(manifest_path) if path not in checksum_names
    ]
    if not missing:
        return
    preview = ", ".join(missing[:8])
    raise ValueError(
        "checksums.sha256 is incomplete versus manifest.json: "
        f"missing {len(missing)} tensor path(s): {preview}"
    )


def verify_checksums(bundle: Path, *, text_only: bool = False) -> int:
    """Hash files listed in `checksums.sha256`.

    Full verification requires every listed file. Text-only verification
    skips a missing file only when that path is outside the text scope.
    A missing manifest, tokenizer, license, or text-family tensor fails.
    """

    checksum_path = bundle / "checksums.sha256"
    if not checksum_path.is_file():
        raise FileNotFoundError(f"Bundle checksum file not found: {checksum_path}")

    verified = 0
    checksum_names: set[str] = set()
    for expected, relative_name in parse_checksum_lines(
        checksum_path.read_text(encoding="utf-8").splitlines()
    ):
        checksum_names.add(relative_name.replace("\\", "/").lstrip("./"))
        path = bundle / relative_name
        if not path.is_file():
            if text_only and not is_text_scope_path(relative_name):
                continue
            raise FileNotFoundError(f"Checksummed bundle file is missing: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        actual = digest.hexdigest()
        if actual != expected:
            raise ValueError(
                f"SHA-256 mismatch for {relative_name}: expected {expected}, got {actual}"
            )
        verified += 1
    if verified == 0:
        raise ValueError("No downloaded files matched checksums.sha256")
    ensure_checksum_list_covers_manifest(bundle, checksum_names)
    return verified


def download_bundle(
    *,
    output: Path,
    repo_id: str,
    revision: str,
    text_only: bool,
) -> tuple[Path, int]:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to download the bundle. "
            "Install it with: python3 -m pip install huggingface_hub"
        ) from exc

    output.mkdir(parents=True, exist_ok=True)
    if text_only:
        ensure_text_only_output_is_clean(output)
    downloaded = Path(
        snapshot_download(
            repo_id,
            repo_type="model",
            revision=revision,
            local_dir=output,
            allow_patterns=patterns_for_scope(text_only=text_only),
        )
    )
    count = verify_checksums(downloaded, text_only=text_only)
    return downloaded, count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="runtime_bundle",
        help="Directory to download into (default: ./runtime_bundle).",
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument(
        "--text-only",
        action="store_true",
        help=(
            "Download the five text-path families plus root files. "
            "Enough for generate(prompt:). generate(image:) also needs "
            "clip_vision and image300m."
        ),
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify an existing directory. Does not contact Hugging Face.",
    )
    args = parser.parse_args(argv)

    output = Path(args.output).expanduser().resolve()
    try:
        if args.verify_only:
            count = verify_checksums(output, text_only=args.text_only)
        else:
            print(f"Downloading {args.repo_id}@{args.revision}")
            print(f"Scope: {'text-only families' if args.text_only else 'full bundle'}")
            output, count = download_bundle(
                output=output,
                repo_id=args.repo_id,
                revision=args.revision,
                text_only=args.text_only,
            )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Verified {count} checksummed files in {output}")
    bpe = output / "bpe_simple_vocab_16e6.txt.gz"
    if bpe.is_file():
        print(f"BPE: {bpe}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
