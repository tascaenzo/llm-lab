import importlib.util
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_extractor():
    path = PROJECT_ROOT / "utils" / "corpus" / "extract_wikipedia.py"
    spec = importlib.util.spec_from_file_location("extract_wikipedia_under_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"impossibile importare {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


extractor = load_extractor()


class CleanWikitextTests(unittest.TestCase):
    def test_nested_links_keep_visible_text(self):
        source = "Prima [[Voce|testo [[Interno|visibile]]]] dopo"
        self.assertEqual(extractor.clean_wikitext(source), "Prima testo visibile dopo")

    def test_multiline_media_and_categories_are_removed(self):
        source = (
            "Prima\n"
            "[[File:Esempio.jpg|thumb|Didascalia\nmultilinea]]\n"
            "[[Categoria:Esempi]]\n"
            "Dopo"
        )
        cleaned = extractor.clean_wikitext(source)
        self.assertEqual(cleaned, "Prima\n\nDopo")
        self.assertNotIn("thumb|", cleaned.lower())

    def test_encoded_and_malformed_references_are_removed(self):
        source = 'Uno &lt;ref name="x"&gt;fonte&lt;/ref&gt; due <refSecondo testo'
        self.assertEqual(extractor.clean_wikitext(source), "Uno due")

    def test_blocks_and_external_links_are_cleaned(self):
        source = (
            'Testo {{tmp|{{nested}}}} {| class="wikitable"\n| dato\n|} '
            "<math>x</math> <code>raw</code> [https://example.com etichetta]"
        )
        self.assertEqual(extractor.clean_wikitext(source), "Testo etichetta")

    def test_preformatted_code_does_not_leak_markup(self):
        source = "Prima\n codice grezzo [[non link]]\nDopo"
        cleaned = extractor.clean_wikitext(source)
        self.assertEqual(cleaned.split(), ["Prima", "Dopo"])
        for marker in extractor.FORBIDDEN_CLEAN_MARKERS:
            self.assertNotIn(marker, cleaned.lower())

    def test_clean_page_rejects_residual_marker(self):
        original_cleaner = extractor.clean_wikitext
        extractor.clean_wikitext = lambda _text: "residuo <ref"
        try:
            with self.assertRaisesRegex(ValueError, "markup residuo"):
                extractor.clean_page(("42", "Titolo", "testo"))
        finally:
            extractor.clean_wikitext = original_cleaner

    def test_safe_name_is_stable(self):
        self.assertEqual(extractor.safe_name(" Italiano Wikipedia V1 "), "italiano-wikipedia-v1")
        with self.assertRaises(ValueError):
            extractor.safe_name("---")

    def test_document_split_matches_dataset_contract(self):
        splits = {
            extractor.document_split(f"wikipedia-it:{page_id}")
            for page_id in range(1, 1000)
        }
        self.assertEqual(splits, {"train", "validation", "test"})
        self.assertEqual(extractor.fnv1a_64("hello"), 0xA430D84680AABD0B)


if __name__ == "__main__":
    unittest.main()
