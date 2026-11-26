"""Checks for scripts/verify_docs.py.

The checker must judge the working tree. It must not ask git whether
verified_commit is an ancestor of HEAD.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import verify_docs  # noqa: E402


class VerifyDocsTests(unittest.TestCase):
    def test_checker_does_not_consult_git_history(self) -> None:
        source = Path(verify_docs.__file__).read_text(encoding="utf-8")
        for forbidden in ("merge-base", "rev-list", "is-ancestor", "subprocess"):
            self.assertNotIn(forbidden, source)

    def test_pin_citation_rejects_the_import_lines(self) -> None:
        line = (
            "`scripts/fetch_weights.py:23-24` pins `signalNotNoise/shap-e-mlx` "
            "at revision `d71f70a09b9c03535188d006a719edda5c19a6eb`."
        )
        match = next(verify_docs.CITATION.finditer(line))
        tokens = verify_docs.adjacent_source_tokens(line, match.start(), match.end(), len(line), None)
        self.assertIn("signalNotNoise/shap-e-mlx", tokens)
        imports = "import sys\nfrom pathlib import Path"
        self.assertFalse(any(token in imports for token in tokens))
        pins = (
            'DEFAULT_REPO_ID = "signalNotNoise/shap-e-mlx"\n'
            'DEFAULT_REVISION = "d71f70a09b9c03535188d006a719edda5c19a6eb"'
        )
        self.assertTrue(any(token in pins for token in tokens))


if __name__ == "__main__":
    unittest.main()
