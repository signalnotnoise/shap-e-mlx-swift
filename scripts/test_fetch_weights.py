"""Verifier tests for scripts/fetch_weights.py.

The text-only case below is the reported failure: a directory whose only
checksummed payload is README.md, with the manifest, tokenizer, and text
tensors absent, must not exit 0.
"""

from __future__ import annotations

import hashlib
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fetch_weights  # noqa: E402


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _line(payload: bytes, name: str) -> str:
    return f"{_digest(payload)}  {name}"


class TextOnlyVerificationTests(unittest.TestCase):
    def test_text_only_fails_when_only_readme_is_present(self) -> None:
        readme = b"readme-only"
        checksum = "\n".join(
            [
                _line(readme, "README.md"),
                f"{'ab' * 32}  manifest.json",
                f"{'cd' * 32}  bpe_simple_vocab_16e6.txt.gz",
                f"{'ef' * 32}  clip_text/token_embedding.bin",
                f"{'12' * 32}  clip_vision/conv.bin",
            ]
        ) + "\n"
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary)
            (bundle / "README.md").write_bytes(readme)
            (bundle / "checksums.sha256").write_text(checksum, encoding="utf-8")
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = fetch_weights.main(
                    ["--verify-only", "--text-only", "--output", str(bundle)]
                )

        self.assertNotEqual(code, 0)
        self.assertIn("manifest.json", stderr.getvalue())
        self.assertNotIn("Verified", stdout.getvalue())

    def test_text_only_skips_missing_files_outside_the_text_scope(self) -> None:
        files = {
            "README.md": b"readme",
            "manifest.json": b"manifest",
            "bpe_simple_vocab_16e6.txt.gz": b"bpe",
            "provenance.json": b"provenance",
            "LICENSE": b"license",
            "licenses/CLIP.txt": b"clip-license",
            "clip_text/token_embedding.bin": b"tensor",
        }
        lines = [_line(payload, name) for name, payload in files.items()]
        lines.append(f"{'12' * 32}  clip_vision/conv.bin")
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary)
            for name, payload in files.items():
                path = bundle / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
            (bundle / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                text_only = fetch_weights.main(
                    ["--verify-only", "--text-only", "--output", str(bundle)]
                )
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                full = fetch_weights.main(["--verify-only", "--output", str(bundle)])

        self.assertEqual(text_only, 0, stderr.getvalue())
        self.assertIn("Verified 7 checksummed files", stdout.getvalue())
        self.assertNotEqual(full, 0)


if __name__ == "__main__":
    unittest.main()
