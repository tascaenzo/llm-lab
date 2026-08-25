#!/usr/bin/env python3

import hashlib
import json
import subprocess
import struct
import sys
import tempfile
from pathlib import Path


MODEL_HEADER = struct.Struct("<8sIIIIQ32s")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if len(sys.argv) != 5:
        raise SystemExit("uso: test_artifact_pipeline.py SCRIPT CLI CORPUS_A CORPUS_B")

    training_script = Path(sys.argv[1])
    cli = Path(sys.argv[2])
    inputs = [Path(sys.argv[3]).resolve(), Path(sys.argv[4]).resolve()]

    with tempfile.TemporaryDirectory() as directory:
        temporary_root = Path(directory)
        manifest_path = temporary_root / "manifest.json"
        manifest = {
            "schema": "llm-lab-test-corpus-v1",
            "name": "fixture-corpus",
            "outputs": {
                "tokenizer_input": [str(path) for path in inputs],
                "tokenizer_input_split": "train",
            },
        }
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        model_path = temporary_root / "artifact.llmtok"
        command = [
            sys.executable,
            str(training_script),
            "--corpus-manifest",
            str(manifest_path),
            "--trainer",
            str(cli),
            "--output",
            str(model_path),
            "--vocab-size",
            "264",
            "--evaluation-bytes",
            "64",
        ]
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            raise AssertionError(f"pipeline fallita:\n{result.stdout}\n{result.stderr}")

        metadata_path = model_path.with_suffix(".llmtok.json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata["schema"] != "llm-lab-tokenizer-artifact-v2":
            raise AssertionError(metadata)
        if metadata["tokenizer"]["sha256"] != sha256(model_path):
            raise AssertionError("checksum del tokenizer non coerente")
        if metadata["trainer"]["sha256"] != sha256(cli):
            raise AssertionError("checksum del trainer non coerente")
        if metadata["corpus"]["manifest_sha256"] != sha256(manifest_path):
            raise AssertionError("checksum del manifest non coerente")
        if metadata["corpus"]["tokenizer_input_split"] != "train":
            raise AssertionError("split tokenizer non registrato")
        if metadata["corpus"]["snapshot"] != manifest:
            raise AssertionError("snapshot del corpus non coerente")
        if metadata["evaluation"]["round_trip"] is not True:
            raise AssertionError("round-trip non riuscito")
        if metadata["tokenizer"]["actual_vocabulary_size"] != 264:
            raise AssertionError("dimensione del vocabolario inattesa")
        model_data = model_path.read_bytes()
        magic, version, header_size, base_size, merge_count, payload_size, checksum = (
            MODEL_HEADER.unpack_from(model_data)
        )
        payload = model_data[header_size:]
        if (magic, version, header_size, base_size) != (b"LLMTOK\r\n", 1, 64, 256):
            raise AssertionError("header binario non valido")
        if merge_count != 8 or payload_size != len(payload):
            raise AssertionError("dimensione del payload non valida")
        if hashlib.sha256(payload).digest() != checksum:
            raise AssertionError("checksum interno del payload non valido")
        if list(temporary_root.glob("*.part")):
            raise AssertionError("la pipeline ha lasciato file parziali")

        original_model_hash = sha256(model_path)
        repeated = subprocess.run(command, check=False, capture_output=True, text=True)
        if repeated.returncode == 0 or "esiste gia'" not in repeated.stderr:
            raise AssertionError("un artefatto esistente dovrebbe essere rifiutato")
        if sha256(model_path) != original_model_hash:
            raise AssertionError("l'artefatto esistente e' stato modificato")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
