import hashlib
import importlib.util
import struct
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_training_script():
    path = PROJECT_ROOT / "utils" / "corpus" / "train_tokenizer.py"
    spec = importlib.util.spec_from_file_location("train_tokenizer_under_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"impossibile importare {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


trainer = load_training_script()


class TrainingMetadataTests(unittest.TestCase):
    def test_model_merge_count_accepts_valid_model(self):
        payload = struct.pack("<IIII", 97, 98, 256, 99)
        header = trainer.MODEL_HEADER.pack(
            trainer.MODEL_MAGIC,
            trainer.MODEL_FORMAT_VERSION,
            trainer.MODEL_HEADER.size,
            trainer.BASE_VOCABULARY_SIZE,
            2,
            len(payload),
            hashlib.sha256(payload).digest(),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "valid.llmtok"
            path.write_bytes(header + payload)
            self.assertEqual(trainer.model_merge_count(path), 2)

    def test_model_merge_count_rejects_inconsistent_models(self):
        invalid_models = (
            b"wrong",
            trainer.MODEL_HEADER.pack(
                b"WRONG!!!",
                1,
                trainer.MODEL_HEADER.size,
                256,
                0,
                0,
                hashlib.sha256(b"").digest(),
            ),
            trainer.MODEL_HEADER.pack(
                trainer.MODEL_MAGIC,
                1,
                trainer.MODEL_HEADER.size,
                256,
                1,
                8,
                hashlib.sha256(b"").digest(),
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.llmtok"
            for content in invalid_models:
                path.write_bytes(content)
                with self.assertRaises(RuntimeError):
                    trainer.model_merge_count(path)

    def test_sha256_file_reads_binary_data(self):
        content = b"tokenizer\x00italiano\xff"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.bin"
            path.write_bytes(content)
            self.assertEqual(trainer.sha256_file(path), hashlib.sha256(content).hexdigest())

    def test_project_relative_keeps_repository_paths_portable(self):
        path = PROJECT_ROOT / "artifacts" / "tokenizers" / "model.llmtok"
        self.assertEqual(
            trainer.project_relative(path, PROJECT_ROOT),
            "artifacts/tokenizers/model.llmtok",
        )


if __name__ == "__main__":
    unittest.main()
