#!/usr/bin/env python3

import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    if len(sys.argv) != 4:
        raise SystemExit("uso: test_experiment.py CLI EXPERIMENT CORPUS")

    cli = Path(sys.argv[1])
    experiment = Path(sys.argv[2])
    corpus = Path(sys.argv[3])

    with tempfile.TemporaryDirectory() as directory:
        model = Path(directory) / "byte-model.llmtok"
        training = subprocess.run(
            [str(cli), "tokenizer", "train", str(model), "256", str(corpus)],
            check=False,
            capture_output=True,
            text=True,
        )
        if training.returncode != 0:
            raise AssertionError(f"training fallito:\n{training.stdout}\n{training.stderr}")

        session = subprocess.run(
            [str(experiment), str(model)],
            input="1\nciao\n2\n99 105 97 111\n2\n-1\n0\n",
            check=False,
            capture_output=True,
            text=True,
        )
        if session.returncode != 0:
            raise AssertionError(f"sessione fallita:\n{session.stdout}\n{session.stderr}")
        if "Token IDs (4): 99 105 97 111" not in session.stdout:
            raise AssertionError(f"codifica inattesa: {session.stdout!r}")
        if "Testo decodificato: ciao" not in session.stdout:
            raise AssertionError(f"decodifica inattesa: {session.stdout!r}")
        if "La lista di ID non e' valida" not in session.stderr:
            raise AssertionError("gli ID negativi dovrebbero essere rifiutati")

        missing_model = subprocess.run(
            [str(experiment)], check=False, capture_output=True, text=True
        )
        if missing_model.returncode == 0 or "MODEL.llmtok" not in missing_model.stderr:
            raise AssertionError("il percorso del modello dovrebbe essere obbligatorio")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
