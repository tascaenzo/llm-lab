#!/usr/bin/env python3
"""Costruisce il corpus multi-sorgente e il tokenizer, guidando la scelta da CLI.

La catena ha nove stadi e ognuno produce file che il successivo consuma. Farla a
mano significa ricordare l'ordine, i percorsi e le opzioni; qui si scelgono le
fonti e il resto segue.

Ogni stadio dichiara i propri output: se esistono gia' viene saltato, quindi lo
script si puo' rilanciare dopo un download interrotto o un errore senza rifare
il lavoro gia' fatto. Niente viene cancellato senza chiederlo.

    python3 utils/corpus/build_corpus.py

Usa soltanto la libreria standard. Lo stadio di normalizzazione di FineWeb-2
richiede pyarrow e lo verifica prima di iniziare.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "utils" / "corpus"
DEFAULT_NAME = "italiano-v3"
RESERVED_TOKENS = 7
PRIORITY = ["wikipedia-it", "wikisource-it", "gutenberg-ita", "fineweb2-ita"]


@dataclass
class Stage:
    key: str
    title: str
    command: list[str]
    outputs: list[Path]
    note: str = ""
    interpreter: str = "python"
    inputs: list[Path] = field(default_factory=list)

    def satisfied(self) -> bool:
        return bool(self.outputs) and all(path.exists() for path in self.outputs)


def ask(question: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        answer = input(f"{question}{suffix}: ").strip()
    except EOFError:
        answer = ""
    return answer or default


def ask_yes(question: str, default: bool = True) -> bool:
    marker = "S/n" if default else "s/N"
    answer = ask(f"{question} ({marker})").lower()
    if not answer:
        return default
    return answer.startswith("s") or answer.startswith("y")


def ask_int(question: str, default: int, minimum: int = 1) -> int:
    while True:
        raw = ask(question, str(default))
        try:
            value = int(raw.replace("_", "").replace(".", ""))
        except ValueError:
            print("  Serve un numero intero.")
            continue
        if value < minimum:
            print(f"  Serve almeno {minimum}.")
            continue
        return value


def human(size: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TiB"


def run(stage: Stage, python: str) -> bool:
    command = [python] + stage.command if stage.interpreter == "python" else stage.command
    print(f"\n\033[1m▶ {stage.title}\033[0m")
    print(f"  $ {' '.join(str(part) for part in command)}")
    started = time.monotonic()
    result = subprocess.run([str(part) for part in command], cwd=ROOT)
    elapsed = time.monotonic() - started
    if result.returncode != 0:
        print(f"\n  \033[31mFallito\033[0m dopo {elapsed:.0f}s: {stage.title}", file=sys.stderr)
        return False
    missing = [path for path in stage.outputs if not path.exists()]
    if missing:
        print(
            f"\n  \033[31mLo stadio e' terminato ma manca l'output atteso:\033[0m "
            f"{missing[0]}",
            file=sys.stderr,
        )
        return False
    print(f"  completato in {elapsed:.0f}s")
    return True


def build_plan(options: argparse.Namespace) -> list[Stage]:
    name = options.name
    clean = ROOT / "data" / "clean" / name
    derived = ROOT / "data" / "derived" / name
    raw_wikipedia = ROOT / "data" / "raw" / "wikipedia-it"
    raw_fineweb = ROOT / "data" / "raw" / "fineweb2-ita"
    binary = ROOT / options.binary
    tokenizer = ROOT / "artifacts" / "tokenizers" / f"{name}.llmtok"
    stages: list[Stage] = []
    dedup_inputs: list[str] = []

    if "wikipedia-it" in options.sources:
        wikipedia_documents = ROOT / "data" / "clean" / "italiano-wikipedia-v1" / "documents.jsonl"
        stages.append(
            Stage(
                "download-wikipedia",
                "Scarico il dump di Wikipedia italiano",
                [CORPUS / "download_wikipedia.py"],
                [raw_wikipedia],
                "circa 4 GB compressi",
            )
        )
        stages.append(
            Stage(
                "extract-wikipedia",
                "Estraggo i documenti da Wikipedia",
                [CORPUS / "extract_wikipedia.py", "--name", "italiano-wikipedia-v1"],
                [wikipedia_documents],
                "usa tutti i core meno uno",
            )
        )
        dedup_inputs.append(f"wikipedia-it={wikipedia_documents}")

    if "wikisource-it" in options.sources:
        raw_wikisource = ROOT / "data" / "raw" / "wikisource-it"
        wikisource_documents = ROOT / "data" / "clean" / "italiano-wikisource-v1" / "documents.jsonl"
        stages.append(
            Stage(
                "download-wikisource",
                "Scarico il dump di Wikisource italiano",
                [CORPUS / "download_wikisource.py", "--destination", raw_wikisource],
                [raw_wikisource],
                "circa 420 MiB",
            )
        )
        stages.append(
            Stage(
                "extract-wikisource",
                "Estraggo i testi da Wikisource",
                [
                    CORPUS / "extract_wikipedia.py",
                    "--name",
                    "italiano-wikisource-v1",
                    "--source",
                    "wikisource-it",
                    "--license",
                    "Wikisource per-page license; verify provenance",
                    "--url-template",
                    "https://it.wikisource.org/?curid={page_id}",
                    "--namespace",
                    "108",
                ],
                [wikisource_documents],
                "stesso estrattore di Wikipedia, provenienza diversa",
            )
        )
        dedup_inputs.append(f"wikisource-it={wikisource_documents}")

    if "gutenberg-ita" in options.sources:
        raw_gutenberg = ROOT / "data" / "raw" / "gutenberg-ita"
        gutenberg_documents = clean / "gutenberg-ita.jsonl"
        stages.append(
            Stage(
                "download-gutenberg",
                f"Scarico fino a {options.gutenberg_books} libri da Project Gutenberg",
                [
                    CORPUS / "download_gutenberg.py",
                    "--limit",
                    str(options.gutenberg_books),
                    "--destination",
                    raw_gutenberg,
                    "--allowlist",
                    options.gutenberg_allowlist,
                ],
                [raw_gutenberg / "catalog.json"],
                "un libro alla volta, con pausa per il mirror",
            )
        )
        stages.append(
            Stage(
                "normalize-gutenberg",
                "Normalizzo i libri e rimuovo la licenza Gutenberg",
                [
                    CORPUS / "normalize_gutenberg.py",
                    "--input",
                    raw_gutenberg,
                    "--output",
                    gutenberg_documents,
                ],
                [gutenberg_documents],
                "i libri vengono spezzati in sezioni",
            )
        )
        dedup_inputs.append(f"gutenberg-ita={gutenberg_documents}")

    if "fineweb2-ita" in options.sources:
        fineweb_documents = clean / "fineweb2-ita.jsonl"
        stages.append(
            Stage(
                "download-fineweb",
                f"Scarico {options.shards} shard di FineWeb-2 italiano",
                [
                    CORPUS / "download_fineweb.py",
                    "--shards",
                    str(options.shards),
                    "--destination",
                    raw_fineweb,
                ],
                [raw_fineweb / "source.json"],
                f"circa {options.shards * 4.5:.0f} GiB",
            )
        )
        stages.append(
            Stage(
                "normalize-fineweb",
                "Normalizzo FineWeb-2 nello schema del corpus",
                [
                    CORPUS / "normalize_source.py",
                    "--source",
                    "fineweb2-ita",
                    "--input",
                    raw_fineweb,
                    "--output",
                    fineweb_documents,
                ],
                [fineweb_documents],
                "richiede pyarrow",
            )
        )
        dedup_inputs.append(f"fineweb2-ita={fineweb_documents}")

    deduplicated = clean / "deduplicated.jsonl"
    dedup_command = [CORPUS / "deduplicate_corpus.py"]
    for entry in dedup_inputs:
        dedup_command += ["--input", entry]
    dedup_command += [
        "--output",
        deduplicated,
        "--report",
        clean / "deduplication.json",
        "--overlap-threshold",
        str(options.overlap_threshold),
    ]
    stages.append(
        Stage(
            "deduplicate",
            "Deduplico fra le fonti",
            dedup_command,
            [deduplicated],
            "le fonti sono in ordine di priorita' decrescente",
        )
    )

    # Il mixer vuole un file per fonte: la deduplicazione ne produce uno solo.
    split_command = [
        CORPUS / "split_by_source.py",
        "--input",
        deduplicated,
        "--output-dir",
        clean / "by-source",
    ]
    per_source = [clean / "by-source" / f"{source}.jsonl" for source in options.sources]
    stages.append(
        Stage(
            "split-by-source",
            "Separo il deduplicato per fonte",
            split_command,
            per_source,
        )
    )

    documents = clean / "documents.jsonl"
    manifest = clean / "manifest.json"
    mix_command = [CORPUS / "mix_corpus.py"]
    for source in options.sources:
        mix_command += ["--input", f"{source}={clean / 'by-source' / f'{source}.jsonl'}"]
        mix_command += ["--quota", f"{source}={options.quotas[source]}"]
    mix_command += [
        "--output",
        documents,
        "--manifest",
        manifest,
        "--name",
        name,
        "--target-tokens",
        str(options.target_tokens),
        "--tokenizer",
        options.calibration_tokenizer,
        "--llm-lab",
        binary,
    ]
    stages.append(
        Stage("mix", "Compongo il corpus con le quote richieste", mix_command, [documents, manifest])
    )

    tokenizer_input = derived / "tokenizer-train-input"
    tokenizer_manifest = derived / "tokenizer-train-input.manifest.json"
    stages.append(
        Stage(
            "tokenizer-corpus",
            "Preparo il corpus di training del tokenizer (solo split train)",
            [
                CORPUS / "prepare_tokenizer_corpus.py",
                "--corpus-manifest",
                manifest,
                "--output-dir",
                tokenizer_input,
                "--output-manifest",
                tokenizer_manifest,
            ],
            [tokenizer_manifest],
            "validation e test restano fuori",
        )
    )
    stages.append(
        Stage(
            "train-tokenizer",
            f"Addestro il tokenizer a {options.vocabulary_size} token",
            [
                CORPUS / "train_tokenizer.py",
                "--corpus-manifest",
                tokenizer_manifest,
                "--trainer",
                binary,
                "--output",
                tokenizer,
                "--vocab-size",
                str(options.vocabulary_size),
            ],
            [tokenizer],
        )
    )

    dataset_prefix = derived / "lm" / name
    stages.append(
        Stage(
            "dataset",
            f"Genero gli split .llmdat con {RESERVED_TOKENS} identificatori riservati",
            [
                binary,
                "dataset",
                "prepare",
                tokenizer,
                documents,
                dataset_prefix,
                "--reserved-tokens",
                str(RESERVED_TOKENS),
            ],
            [Path(f"{dataset_prefix}.train.llmdat")],
            "il vocabolario del modello resta congelato da qui in poi",
            interpreter="binary",
        )
    )
    return stages


def interactive_options(defaults: argparse.Namespace) -> argparse.Namespace:
    print("\n\033[1mCostruzione del corpus multi-sorgente\033[0m\n")
    sources = []
    if ask_yes("Includere Wikipedia italiano (enciclopedico)?"):
        sources.append("wikipedia-it")
    if ask_yes("Includere Wikisource italiano (narrativo, pubblico dominio)?"):
        sources.append("wikisource-it")
    if ask_yes(
        "Includere Project Gutenberg italiano (richiede una allowlist di diritti verificata)?",
        default=False,
    ):
        sources.append("gutenberg-ita")
    if ask_yes("Includere FineWeb-2 italiano (web, richiede pyarrow)?"):
        sources.append("fineweb2-ita")
    if not sources:
        print("Nessuna fonte selezionata.", file=sys.stderr)
        raise SystemExit(1)
    defaults.sources = [source for source in PRIORITY if source in sources]

    if "gutenberg-ita" in sources:
        print("\n  Il catalogo italiano di Gutenberg contiene circa 1.100 libri.")
        defaults.gutenberg_books = ask_int("  Quanti libri scaricare al massimo", 1200)
    if "fineweb2-ita" in sources:
        print("\n  Una shard di FineWeb-2 pesa circa 4,5 GiB e rende circa 2,6 miliardi di token.")
        defaults.shards = ask_int("  Quante shard scaricare", 1)

    billions = ask("\nToken totali desiderati, in miliardi", "3")
    try:
        defaults.target_tokens = int(float(billions.replace(",", ".")) * 1e9)
    except ValueError:
        print("Valore non valido.", file=sys.stderr)
        raise SystemExit(1)

    quotas = {}
    if len(defaults.sources) == 1:
        quotas[defaults.sources[0]] = 1.0
    else:
        print("\n  Quote per fonte, in percentuale sul totale dei token.")
        remaining = 100
        for index, source in enumerate(defaults.sources):
            if index == len(defaults.sources) - 1:
                quotas[source] = remaining / 100
                print(f"  {source}: {remaining}% (il resto)")
            else:
                default = {"wikipedia-it": 25, "wikisource-it": 15, "gutenberg-ita": 10}.get(
                    source, 60
                )
                value = ask_int(f"  {source} (%)", min(default, remaining), minimum=0)
                value = min(value, remaining)
                quotas[source] = value / 100
                remaining -= value
    defaults.quotas = quotas
    defaults.vocabulary_size = ask_int("\nDimensione del vocabolario", 32000)
    return defaults


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default=DEFAULT_NAME, help="nome del corpus")
    parser.add_argument("--binary", default="build/release/llm-lab", help="binario llm-lab")
    parser.add_argument(
        "--calibration-tokenizer",
        default="artifacts/tokenizers/italiano-wikipedia-v2.llmtok",
        help="tokenizer esistente, usato solo per stimare i token nelle quote",
    )
    parser.add_argument("--overlap-threshold", type=float, default=0.5)
    parser.add_argument(
        "--gutenberg-allowlist",
        default="utils/corpus/gutenberg-it-allowlist.json",
        help="allowlist dei titoli Gutenberg verificati per l'Italia",
    )
    parser.add_argument("--dry-run", action="store_true", help="mostra il piano e termina")
    parser.add_argument("--force", action="store_true", help="riesegue anche gli stadi completati")
    options = parser.parse_args()
    options.sources = []
    options.shards = 1
    options.gutenberg_books = 1200
    options.target_tokens = 3_000_000_000
    options.quotas = {}
    options.vocabulary_size = 32000

    binary = ROOT / options.binary
    if not binary.is_file():
        print(
            f"Binario non trovato: {binary}\n"
            "Costruiscilo prima:\n"
            "  cmake --preset release && cmake --build --preset release",
            file=sys.stderr,
        )
        return 1

    options = interactive_options(options)
    if "fineweb2-ita" in options.sources:
        probe = subprocess.run(
            [sys.executable, "-c", "import pyarrow"], capture_output=True
        )
        if probe.returncode != 0:
            print(
                "\npyarrow non e' disponibile per questo interprete, e serve a leggere il\n"
                "Parquet di FineWeb-2. Installalo e rilancia con lo stesso interprete:\n"
                f"  {sys.executable} -m pip install pyarrow",
                file=sys.stderr,
            )
            return 1

    stages = build_plan(options)
    free = shutil.disk_usage(ROOT).free
    print("\n\033[1mPiano\033[0m")
    for index, stage in enumerate(stages, 1):
        state = "\033[32mgia' fatto\033[0m" if stage.satisfied() and not options.force else "da fare"
        note = f"  ({stage.note})" if stage.note else ""
        print(f"  {index}. {stage.title}{note} — {state}")
    print(f"\n  Spazio libero: {human(free)}")
    print(f"  Corpus: {options.name}, {options.target_tokens/1e9:.2f} miliardi di token")
    for source, quota in options.quotas.items():
        print(f"    {source}: {quota:.0%}")

    if options.dry_run:
        return 0
    if not ask_yes("\nProcedo", default=False):
        print("Annullato.")
        return 0

    for stage in stages:
        if stage.satisfied() and not options.force:
            print(f"\n\033[1m▶ {stage.title}\033[0m\n  gia' fatto, salto")
            continue
        if not run(stage, sys.executable):
            print(
                "\nLa catena si e' fermata. Rilancia lo script quando hai risolto:\n"
                "  gli stadi gia' completati verranno saltati.",
                file=sys.stderr,
            )
            return 1

    print("\n\033[1;32mCorpus e tokenizer pronti.\033[0m")
    manifest = ROOT / "data" / "clean" / options.name / "manifest.json"
    if manifest.is_file():
        document = json.loads(manifest.read_text(encoding="utf-8"))
        print(f"  documenti: {document['documents']}")
        print(f"  token stimati: {document['estimated_tokens']/1e9:.3f} miliardi")
        for entry in document["sources"]:
            print(
                f"    {entry['source']:16} quota {entry['requested_quota']:.0%} "
                f"-> {entry['achieved_quota']:.1%}"
            )
    print("\n  Prossimo passo: il pilota di training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
