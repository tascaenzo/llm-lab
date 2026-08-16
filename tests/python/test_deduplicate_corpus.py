"""Verifica della deduplicazione incrociata fra fonti del corpus."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
SCRIPT = REPOSITORY / "utils" / "corpus" / "deduplicate_corpus.py"

ARTICLE = (
    "Il Rinascimento fu un periodo di grande fioritura artistica e culturale in Italia. "
    "Ebbe origine a Firenze nel corso del quattordicesimo secolo e si diffuse in Europa. "
    "Gli artisti riscoprirono le proporzioni e la prospettiva della statuaria classica. "
    "La stampa a caratteri mobili accelero' la circolazione dei testi umanistici. "
    "Le corti signorili finanziarono botteghe, biblioteche e cantieri architettonici."
)

OTHER_ARTICLE = (
    "La transumanza e' lo spostamento stagionale delle greggi fra pascoli di quota diversa. "
    "In Abruzzo i tratturi collegavano le montagne interne alle pianure della Puglia. "
    "I pastori percorrevano quelle vie erbose per settimane, seguendo tappe fissate. "
    "La pratica ha lasciato toponimi, chiese rurali e un diritto consuetudinario proprio. "
    "Oggi sopravvive in forma ridotta ed e' riconosciuta come patrimonio culturale."
)

BOILERPLATE_HEADER = (
    "Home Chi siamo Contatti Newsletter Accedi Registrati. "
    "Benvenuti sul nostro portale dedicato alla cultura italiana e alle sue tradizioni. "
)
BOILERPLATE_FOOTER = (
    " Condividi su Facebook Twitter WhatsApp. Lascia un commento qui sotto. "
    "Tutti i diritti riservati. Informativa sui cookie e sulla privacy del sito."
)


def write_jsonl(path: Path, documents: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for document in documents:
            handle.write(json.dumps(document, ensure_ascii=False) + "\n")


class DeduplicateCorpusTest(unittest.TestCase):
    def run_deduplication(self, wikipedia: list[dict], web: list[dict], **options):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wikipedia_path = root / "wikipedia.jsonl"
            web_path = root / "web.jsonl"
            output = root / "deduped.jsonl"
            report = root / "report.json"
            write_jsonl(wikipedia_path, wikipedia)
            write_jsonl(web_path, web)
            command = [
                sys.executable,
                str(SCRIPT),
                "--input",
                f"wikipedia={wikipedia_path}",
                "--input",
                f"web={web_path}",
                "--output",
                str(output),
                "--report",
                str(report),
            ]
            for key, value in options.items():
                command += [f"--{key.replace('_', '-')}", str(value)]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            kept = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            return kept, json.loads(report.read_text(encoding="utf-8"))

    def test_exact_duplicate_across_sources_is_dropped(self):
        kept, report = self.run_deduplication(
            [{"id": "wikipedia:1", "source": "wikipedia", "text": ARTICLE}],
            [{"id": "web:1", "source": "web", "text": ARTICLE}],
        )
        self.assertEqual([document["id"] for document in kept], ["wikipedia:1"])
        web = next(entry for entry in report["sources"] if entry["source"] == "web")
        self.assertEqual(web["dropped_exact"], 1)
        self.assertEqual(web["dropped_by_source"], {"wikipedia": 1})

    def test_article_wrapped_in_boilerplate_is_dropped(self):
        """Il caso reale: un articolo mirrorato dentro una pagina con menu e footer.

        L'hash del documento intero non lo vede, perche' il testo attorno e'
        diverso. Deve intercettarlo la sovrapposizione di frasi.
        """
        kept, report = self.run_deduplication(
            [{"id": "wikipedia:1", "source": "wikipedia", "text": ARTICLE}],
            [
                {
                    "id": "web:mirror",
                    "source": "web",
                    "text": BOILERPLATE_HEADER + ARTICLE + BOILERPLATE_FOOTER,
                }
            ],
        )
        self.assertEqual([document["id"] for document in kept], ["wikipedia:1"])
        web = next(entry for entry in report["sources"] if entry["source"] == "web")
        self.assertEqual(web["dropped_overlap"], 1)
        self.assertEqual(web["dropped_by_source"], {"wikipedia": 1})

    def test_unrelated_document_is_kept(self):
        kept, report = self.run_deduplication(
            [{"id": "wikipedia:1", "source": "wikipedia", "text": ARTICLE}],
            [{"id": "web:1", "source": "web", "text": OTHER_ARTICLE}],
        )
        self.assertEqual(
            sorted(document["id"] for document in kept), ["web:1", "wikipedia:1"]
        )
        web = next(entry for entry in report["sources"] if entry["source"] == "web")
        self.assertEqual(web["dropped_exact"], 0)
        self.assertEqual(web["dropped_overlap"], 0)

    def test_higher_priority_source_survives_the_collision(self):
        """L'ordine degli argomenti e' l'ordine di priorita': vince chi viene prima."""
        kept, _ = self.run_deduplication(
            [{"id": "wikipedia:1", "source": "wikipedia", "text": ARTICLE}],
            [{"id": "web:1", "source": "web", "text": ARTICLE + BOILERPLATE_FOOTER}],
        )
        self.assertEqual([document["id"] for document in kept], ["wikipedia:1"])

    def test_duplicates_inside_one_source_are_dropped(self):
        kept, report = self.run_deduplication(
            [],
            [
                {"id": "web:1", "source": "web", "text": ARTICLE},
                {"id": "web:2", "source": "web", "text": ARTICLE},
                {"id": "web:3", "source": "web", "text": OTHER_ARTICLE},
            ],
        )
        self.assertEqual(sorted(document["id"] for document in kept), ["web:1", "web:3"])
        web = next(entry for entry in report["sources"] if entry["source"] == "web")
        self.assertEqual(web["dropped_exact"], 1)

    def test_short_documents_use_exact_matching_only(self):
        """Sotto la soglia di frasi la sovrapposizione non si applica.

        Frasi isolate come un titolo o una data ricorrono in pagine non
        imparentate: usarle per dedurre un duplicato scarterebbe testo buono.
        """
        short = "Milano e' il capoluogo della Lombardia ed e' un importante centro economico."
        kept, _ = self.run_deduplication(
            [{"id": "wikipedia:1", "source": "wikipedia", "text": short}],
            [{"id": "web:1", "source": "web", "text": short + " Il clima e' continentale."}],
        )
        self.assertEqual(sorted(document["id"] for document in kept), ["web:1", "wikipedia:1"])


if __name__ == "__main__":
    unittest.main()
