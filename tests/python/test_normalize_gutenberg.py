"""Verifica della normalizzazione dei libri di Project Gutenberg."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
SCRIPT = REPOSITORY / "utils" / "corpus" / "normalize_gutenberg.py"

LICENSE_HEADER = (
    "The Project Gutenberg eBook of Prova\n\n"
    "This eBook is for the use of anyone anywhere in the United States and most\n"
    "other parts of the world at no cost and with almost no restrictions "
    "whatsoever. You may copy it, give it away or re-use it under the terms of\n"
    "the Project Gutenberg License included with this eBook.\n\n"
)
LICENSE_FOOTER = (
    "\n\nSection 1. General Terms of Use and Redistributing Project Gutenberg\n"
    "electronic works. Please read this before you distribute or use this work.\n"
)
BODY = "\n\n".join(
    f"Capitolo {index}. " + ("Una frase di prosa continua che occupa spazio sufficiente. " * 12)
    for index in range(1, 6)
)


def build_source(directory: Path, text: str, identifier: int = 42) -> Path:
    root = directory / "gutenberg"
    (root / "texts").mkdir(parents=True)
    (root / "texts" / f"{identifier}.txt").write_text(text, encoding="utf-8")
    (root / "catalog.json").write_text(
        json.dumps(
            {
                "schema": "llm-lab-source-v1",
                "source": "gutenberg-ita",
                "license": "Public domain (Project Gutenberg)",
                "books": [
                    {
                        "id": identifier,
                        "title": "Prova",
                        "authors": ["Anonimo"],
                        "url": f"https://example.invalid/{identifier}.txt",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return root


def normalize(root: Path, output: Path, **options) -> dict:
    command = [sys.executable, str(SCRIPT), "--input", str(root), "--output", str(output)]
    for key, value in options.items():
        command += [f"--{key.replace('_', '-')}", str(value)]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise AssertionError(result.stderr)
    return json.loads(output.with_suffix(".manifest.json").read_text(encoding="utf-8"))


class NormalizeGutenbergTest(unittest.TestCase):
    def test_modern_markers_remove_the_license(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = build_source(
                root,
                LICENSE_HEADER
                + "*** START OF THE PROJECT GUTENBERG EBOOK PROVA ***\n"
                + BODY
                + "\n*** END OF THE PROJECT GUTENBERG EBOOK PROVA ***"
                + LICENSE_FOOTER,
            )
            output = root / "documents.jsonl"
            manifest = normalize(source, output)
            documents = [
                json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()
            ]
            self.assertTrue(documents)
            self.assertEqual(manifest["statistics"]["books_without_markers"], 0)
            for document in documents:
                self.assertNotIn("Project Gutenberg", document["text"])
                self.assertEqual(document["source"], "gutenberg-ita")
                self.assertTrue(document["id"].startswith("gutenberg:42-"))

    def test_legacy_footer_is_removed_even_before_the_modern_one(self):
        """I file vecchi chiudono con "End of Project Gutenberg's ..." senza asterischi.

        Quando compaiono entrambe le forme vale la piu' a sinistra, altrimenti
        fra le due resta il colophon del trascrittore.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = build_source(
                root,
                "*** START OF THE PROJECT GUTENBERG EBOOK PROVA ***\n"
                + BODY
                + "\n\nEnd of Project Gutenberg's Prova, by Anonimo\n\n"
                + "Nota del trascrittore sulle convenzioni tipografiche adottate.\n"
                + "\n*** END OF THE PROJECT GUTENBERG EBOOK PROVA ***"
                + LICENSE_FOOTER,
            )
            output = root / "documents.jsonl"
            normalize(source, output)
            # L'asserzione va sui testi: il campo license nomina Gutenberg per
            # definizione ed e' corretto che lo faccia.
            documents = [
                json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()
            ]
            self.assertTrue(documents)
            for document in documents:
                self.assertNotIn("Gutenberg", document["text"])
                self.assertNotIn("trascrittore", document["text"])

    def test_book_without_markers_is_discarded(self):
        """Senza marcatori non si distingue il testo dalla licenza."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = build_source(root, LICENSE_HEADER + BODY + LICENSE_FOOTER)
            output = root / "documents.jsonl"
            manifest = normalize(source, output)
            self.assertEqual(manifest["statistics"]["books_without_markers"], 1)
            self.assertEqual(manifest["statistics"]["documents_written"], 0)

    def test_long_book_is_split_into_sections(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = build_source(
                root,
                "*** START OF THE PROJECT GUTENBERG EBOOK PROVA ***\n"
                + BODY
                + "\n*** END OF THE PROJECT GUTENBERG EBOOK PROVA ***",
            )
            output = root / "documents.jsonl"
            normalize(source, output, section_characters=1000, min_characters=100)
            documents = [
                json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()
            ]
            self.assertGreater(len(documents), 1)
            identifiers = [document["id"] for document in documents]
            self.assertEqual(len(identifiers), len(set(identifiers)))


if __name__ == "__main__":
    unittest.main()
