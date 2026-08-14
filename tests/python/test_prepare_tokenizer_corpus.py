import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_module():
    path = PROJECT_ROOT / "utils" / "corpus" / "prepare_tokenizer_corpus.py"
    spec = importlib.util.spec_from_file_location("prepare_tokenizer_corpus_under_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"impossibile importare {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


preparer = load_module()


class PrepareTokenizerCorpusTests(unittest.TestCase):
    def test_fnv_matches_dataset_contract(self):
        self.assertEqual(preparer.fnv1a_64("hello"), 0xA430D84680AABD0B)

    def test_prepare_writes_only_train_documents(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            documents = root / "documents.jsonl"
            records = [
                {"id": f"doc-{index}", "text": f"testo {index}"}
                for index in range(200)
            ]
            documents.write_text(
                "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
            )
            source_manifest = root / "source.json"
            source_manifest.write_text(
                json.dumps({
                    "name": "fixture",
                    "outputs": {"documents": str(documents)},
                }),
                encoding="utf-8",
            )
            output_manifest = root / "derived" / "manifest.json"
            manifest = preparer.prepare(
                source_manifest, root / "parts", output_manifest, 128, PROJECT_ROOT
            )
            expected = [
                record["text"] for record in records
                if preparer.document_split(record["id"]) == "train"
            ]
            actual = "".join(
                path.read_text(encoding="utf-8")
                for path in sorted((root / "parts").glob("part-*.txt"))
            )
            for text in expected:
                self.assertIn(f"{text}\n\n", actual)
            for record in records:
                if preparer.document_split(record["id"]) != "train":
                    self.assertNotIn(f"{record['text']}\n\n", actual)
            self.assertEqual(manifest["outputs"]["tokenizer_input_split"], "train")
            self.assertEqual(manifest["statistics"]["documents"]["train"], len(expected))


if __name__ == "__main__":
    unittest.main()
