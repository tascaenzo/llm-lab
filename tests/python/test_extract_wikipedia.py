"""Regression coverage for namespace-aware MediaWiki extraction."""

from __future__ import annotations

import bz2
import importlib.util
import tempfile
import unittest
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
SCRIPT = REPOSITORY / "utils" / "corpus" / "extract_wikipedia.py"
SPEC = importlib.util.spec_from_file_location("extract_wikipedia", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
extract = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(extract)


class CleanWikitextTests(unittest.TestCase):
    def test_nested_links_keep_visible_text(self):
        source = "Prima [[Voce|testo [[Interno|visibile]]]] dopo"
        self.assertEqual(extract.clean_wikitext(source), "Prima testo visibile dopo")

    def test_multiline_media_and_categories_are_removed(self):
        source = (
            "Prima\n"
            "[[File:Esempio.jpg|thumb|Didascalia\nmultilinea]]\n"
            "[[Categoria:Esempi]]\n"
            "Dopo"
        )
        cleaned = extract.clean_wikitext(source)
        self.assertEqual(cleaned, "Prima\n\nDopo")
        self.assertNotIn("thumb|", cleaned.lower())

    def test_encoded_and_malformed_references_are_removed(self):
        source = 'Uno &lt;ref name="x"&gt;fonte&lt;/ref&gt; due <refSecondo testo'
        self.assertEqual(extract.clean_wikitext(source), "Uno due")

    def test_blocks_and_external_links_are_cleaned(self):
        source = (
            'Testo {{tmp|{{nested}}}} {| class="wikitable"\n| dato\n|} '
            "<math>x</math> <code>raw</code> [https://example.com etichetta]"
        )
        self.assertEqual(extract.clean_wikitext(source), "Testo etichetta")

    def test_preformatted_code_does_not_leak_markup(self):
        source = "Prima\n codice grezzo [[non link]]\nDopo"
        cleaned = extract.clean_wikitext(source)
        self.assertEqual(cleaned.split(), ["Prima", "Dopo"])
        for marker in extract.FORBIDDEN_CLEAN_MARKERS:
            self.assertNotIn(marker, cleaned.lower())

    def test_clean_page_rejects_residual_marker(self):
        original_cleaner = extract.clean_wikitext
        extract.clean_wikitext = lambda _text: "residuo <ref"
        try:
            with self.assertRaisesRegex(ValueError, "markup residuo"):
                extract.clean_page(("42", "Titolo", "testo"))
        finally:
            extract.clean_wikitext = original_cleaner

    def test_safe_name_is_stable(self):
        self.assertEqual(extract.safe_name(" Italiano Wikipedia V1 "), "italiano-wikipedia-v1")
        with self.assertRaises(ValueError):
            extract.safe_name("---")

    def test_document_split_matches_dataset_contract(self):
        splits = {extract.document_split(f"wikipedia-it:{page_id}") for page_id in range(1, 1000)}
        self.assertEqual(splits, {"train", "validation", "test"})
        self.assertEqual(extract.fnv1a_64("hello"), 0xA430D84680AABD0B)


def page(identifier: int, namespace: int, title: str, text: str) -> str:
    return (
        "<page><title>"
        f"{title}</title><ns>{namespace}</ns><id>{identifier}</id>"
        f"<revision><id>{identifier + 100}</id><text>{text}</text></revision></page>"
    )


class NamespaceExtractionTest(unittest.TestCase):
    def test_wikisource_page_namespace_is_selected_without_main_pages(self):
        payload = (
            '<?xml version="1.0"?><mediawiki>'
            + page(1, 0, "Opera", "{{Pagina|Opera/1}}")
            + page(2, 108, "Pagina:Opera/1", "Testo narrativo trascritto.")
            + "</mediawiki>"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.xml.bz2"
            path.write_bytes(bz2.compress(payload.encode("utf-8")))
            statistics = {"pages_seen": 0, "pages_selected_namespace": 0, "documents_written": 0}
            reporter = extract.ProgressReporter(path.stat().st_size)
            pages = list(extract.iter_raw_pages(path, statistics, reporter, 108))

        self.assertEqual(pages, [("2", "Pagina:Opera/1", "Testo narrativo trascritto.")])
        self.assertEqual(statistics["pages_seen"], 2)
        self.assertEqual(statistics["pages_selected_namespace"], 1)


if __name__ == "__main__":
    unittest.main()
