import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_module(name, relative_path):
    path = PROJECT_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"impossibile importare {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


validator = load_module("validate_sft_under_test", "utils/sft/validate_conversations.py")
evaluator = load_module("evaluate_sft_under_test", "utils/sft/evaluate_chat.py")


def identifiers_for_all_splits():
    found = {}
    index = 0
    while len(found) != 3:
        identifier = f"conversation-{index}"
        found.setdefault(validator.split_for(identifier), identifier)
        index += 1
    return found


class SftToolTests(unittest.TestCase):
    def test_validator_accepts_schema_and_reports_all_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "conversations.jsonl"
            records = [
                {
                    "id": identifier,
                    "source": "fixture-v1",
                    "license": "CC0-1.0",
                    "messages": [
                        {"role": "user", "content": f"domanda {split}"},
                        {"role": "assistant", "content": "risposta"},
                    ],
                }
                for split, identifier in identifiers_for_all_splits().items()
            ]
            path.write_text(
                "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
            )
            report = validator.validate(path, None)
            self.assertTrue(report["valid"])
            self.assertEqual(report["examples"], 3)
            self.assertEqual(report["assistant_turns"], 3)
            self.assertEqual(set(report["splits"]), {"train", "validation", "test"})

    def test_validator_rejects_duplicate_identifier(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "conversations.jsonl"
            record = {
                "id": "duplicate",
                "source": "fixture-v1",
                "license": "CC0-1.0",
                "messages": [
                    {"role": "user", "content": "domanda"},
                    {"role": "assistant", "content": "risposta"},
                ],
            }
            path.write_text((json.dumps(record) + "\n") * 2, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicato"):
                validator.validate(path, None)

    def test_evaluation_suite_rejects_duplicate_identifier(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "suite.jsonl"
            record = {"id": "same", "prompt": "Una domanda"}
            path.write_text((json.dumps(record) + "\n") * 2, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicato"):
                evaluator.load_suite(path)


if __name__ == "__main__":
    unittest.main()
