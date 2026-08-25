"""Rights gates for the optional Gutenberg source."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPOSITORY = Path(__file__).resolve().parents[2]
SCRIPT = REPOSITORY / "utils" / "corpus" / "download_gutenberg.py"
SPEC = importlib.util.spec_from_file_location("download_gutenberg", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
gutenberg = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gutenberg)


class GutenbergRightsTest(unittest.TestCase):
    def test_allowlist_requires_a_nonempty_italian_rights_record(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "allowlist.json"
            path.write_text(
                json.dumps(
                    {
                        "schema": "llm-lab-gutenberg-allowlist-v1",
                        "jurisdiction": "IT",
                        "books": [
                            {
                                "id": 42,
                                "rights": "public-domain-it",
                                "evidence": "Archivio verificato",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            allowed = gutenberg.load_allowlist(path)
            self.assertEqual(allowed[42]["rights"], "public-domain-it")

            path.write_text(
                json.dumps(
                    {
                        "schema": "llm-lab-gutenberg-allowlist-v1",
                        "jurisdiction": "IT",
                        "books": [],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                gutenberg.load_allowlist(path)

    def test_catalog_requires_both_the_allowlist_and_uncopyrighted_metadata(self):
        payload = {
            "results": [
                {"id": 1, "copyright": False, "title": "Consentito", "authors": []},
                {"id": 2, "copyright": False, "title": "Non verificato", "authors": []},
                {"id": 3, "copyright": True, "title": "Copyright", "authors": []},
            ],
            "next": None,
        }
        allowlist = {
            1: {"rights": "public-domain-it", "evidence": "Prova"},
            3: {"rights": "public-domain-it", "evidence": "Prova"},
        }
        with patch.object(gutenberg, "fetch", return_value=json.dumps(payload).encode("utf-8")):
            books = gutenberg.load_catalog(10, allowlist)
        self.assertEqual([book["id"] for book in books], [1])


if __name__ == "__main__":
    unittest.main()
