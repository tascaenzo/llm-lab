#!/usr/bin/env python3
"""Generate a deterministic SFT sanity probe for exact factual recall.

This is deliberately *not* a production corpus.  Every training prompt is
tested after training, so failure demonstrates an issue in the SFT setup or in
the base model's ability to bind a simple prompt to an answer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TRAIN_FACTS = [
    ("capitale-italia", "Qual è la capitale d'Italia?", "La capitale d'Italia è Roma."),
    ("capitale-francia", "Qual è la capitale della Francia?", "La capitale della Francia è Parigi."),
    ("capitale-spagna", "Qual è la capitale della Spagna?", "La capitale della Spagna è Madrid."),
    ("formula-acqua", "Qual è la formula chimica dell'acqua?", "La formula chimica dell'acqua è H₂O."),
    ("pianeta-rosso", "Quale pianeta è chiamato Pianeta Rosso?", "Il Pianeta Rosso è Marte."),
    ("pianeta-sole", "Qual è il pianeta più vicino al Sole?", "Il pianeta più vicino al Sole è Mercurio."),
    ("sole-stella", "Il Sole è un pianeta o una stella?", "Il Sole è una stella."),
    ("moltiplicazione", "Quanto fa 7 per 8?", "Sette per otto fa 56."),
    ("percentuale", "Quanto fa il 10 per cento di 250?", "Il 10 per cento di 250 è 25."),
    ("frazione", "Quanto fa un quarto di 100?", "Un quarto di 100 è 25."),
    ("dante", "Chi ha scritto la Divina Commedia?", "La Divina Commedia è stata scritta da Dante Alighieri."),
    ("manzoni", "Chi ha scritto I promessi sposi?", "I promessi sposi è un romanzo di Alessandro Manzoni."),
    ("gioconda", "Chi ha dipinto la Gioconda?", "La Gioconda è attribuita a Leonardo da Vinci."),
    ("ram", "Che differenza c'è tra RAM e archiviazione?", "La RAM conserva dati temporanei; l'archiviazione conserva file e app anche a dispositivo spento."),
    ("browser", "Che cos'è un browser?", "Un browser è un programma usato per visitare siti e servizi sul web."),
    ("backup", "Che cos'è un backup?", "Un backup è una copia di sicurezza dei dati."),
    ("perimetro", "Come si calcola il perimetro di un rettangolo?", "Il perimetro è due volte la base più due volte l'altezza."),
    ("area", "Come si calcola l'area di un rettangolo?", "L'area si calcola moltiplicando base per altezza."),
    ("sinonimo", "Che cos'è un sinonimo?", "Un sinonimo è una parola con significato uguale o molto simile a un'altra parola."),
    ("verbo", "Che cos'è un verbo?", "Un verbo è una parola che esprime un'azione, uno stato o un evento."),
]

VALIDATION_FACTS = [
    ("capitale-portogallo", "Qual è la capitale del Portogallo?", "La capitale del Portogallo è Lisbona."),
    ("ossigeno", "Qual è il simbolo chimico dell'ossigeno?", "Il simbolo chimico dell'ossigeno è O."),
    ("muro-berlino", "In quale anno cadde il Muro di Berlino?", "Il Muro di Berlino cadde nel 1989."),
    ("media", "Come si calcola la media aritmetica?", "Si sommano i valori e si divide per il loro numero."),
]

TEST_FACTS = [
    ("capitale-canada", "Qual è la capitale del Canada?", "La capitale del Canada è Ottawa."),
    ("oro", "Qual è il simbolo chimico dell'oro?", "Il simbolo chimico dell'oro è Au."),
    ("po", "Qual è il fiume più lungo d'Italia?", "Il fiume più lungo d'Italia è il Po."),
]


def fnv1a_64(text: str) -> int:
    value = 14695981039346656037
    for byte in text.encode("utf-8"):
        value ^= byte
        value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return value


def id_for_split(label: str, target: str) -> str:
    for nonce in range(100_000):
        candidate = f"local-curated:sft-sanity-facts-v1:{label}:{nonce}"
        bucket = fnv1a_64(candidate) % 10_000
        actual = "train" if bucket < 9000 else "validation" if bucket < 9500 else "test"
        if actual == target:
            return candidate
    raise RuntimeError(f"could not find a {target} ID for {label}")


def make_record(label: str, question: str, answer: str, target_split: str) -> dict:
    return {
        "id": id_for_split(label, target_split),
        "source": "local-curated:sft-sanity-facts-v1",
        "license": "USER_APPROVED",
        "approval_status": "approved",
        "category": "sft_sanity_probe",
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        "provenance": {
            "kind": "sft_exact_recall_sanity_probe",
            "generator": "utils/sft/generate_sft_sanity_facts_v1.py",
            "intended_split": target_split,
            "not_for_production_training": True,
        },
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8") as destination:
        for row in rows:
            destination.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    output_directory = args.output_directory
    source_path = output_directory / "sft-sanity-facts-v1.source.jsonl"
    prompts_path = output_directory / "sft-sanity-facts-v1.manual-tests.jsonl"
    outputs = (source_path, prompts_path)
    existing = [path for path in outputs if path.exists()]
    if existing:
        raise SystemExit(f"refusing to overwrite existing output: {existing[0]}")

    train_rows = [make_record(*fact, "train") for fact in TRAIN_FACTS]
    validation_rows = [make_record(*fact, "validation") for fact in VALIDATION_FACTS]
    test_rows = [make_record(*fact, "test") for fact in TEST_FACTS]
    output_directory.mkdir(parents=True, exist_ok=True)
    write_jsonl(source_path, train_rows + validation_rows + test_rows)
    write_jsonl(prompts_path, train_rows)
    print(json.dumps({
        "output_directory": str(output_directory),
        "train_examples": len(train_rows),
        "validation_examples": len(validation_rows),
        "test_examples": len(test_rows),
        "manual_test_examples": len(train_rows),
    }))


if __name__ == "__main__":
    main()
