import os
import tempfile
import unittest
from pathlib import Path

from utils.benchmarks.project_environment import configured_backend, read_dotenv


class ProjectEnvironmentTests(unittest.TestCase):
    def test_reads_quotes_comments_and_export(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text(
                "# local settings\n"
                "export LLM_LAB_BACKEND='metal' # selected accelerator\n"
                'RUNPOD_GPU_ID="NVIDIA GeForce RTX 4090"\n',
                encoding="utf-8",
            )

            values = read_dotenv(path)

            self.assertEqual(values["LLM_LAB_BACKEND"], "metal")
            self.assertEqual(values["RUNPOD_GPU_ID"], "NVIDIA GeForce RTX 4090")

    def test_process_environment_precedes_dotenv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".env").write_text("LLM_LAB_BACKEND=metal\n", encoding="utf-8")

            backend = configured_backend(
                root,
                default="cpu",
                allowed=("cpu", "metal", "cuda"),
                environment={"LLM_LAB_BACKEND": "cuda"},
            )

            self.assertEqual(backend, "cuda")

    def test_explicit_missing_file_is_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.env"
            with self.assertRaisesRegex(ValueError, "does not exist"):
                configured_backend(
                    Path(directory),
                    default="cpu",
                    allowed=("cpu", "metal", "cuda"),
                    environment={"LLM_LAB_ENV_FILE": os.fspath(missing)},
                )


if __name__ == "__main__":
    unittest.main()
