#!/usr/bin/env python3
"""Create a small, traceable Italian factual-QA seed corpus.

The default output contains only evergreen facts.  Current office holders are
optional because an SFT model cannot update them by itself: those records carry
an explicit ``valid_at`` date and must not be mixed into a timeless corpus.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EVERGREEN_FACTS = [
    ("geografia-capitale-italia", "Qual è la capitale d'Italia?", "La capitale d'Italia è Roma.", "geografia"),
    ("geografia-capitale-francia", "Qual è la capitale della Francia?", "La capitale della Francia è Parigi.", "geografia"),
    ("geografia-capitale-spagna", "Qual è la capitale della Spagna?", "La capitale della Spagna è Madrid.", "geografia"),
    ("geografia-capitale-germania", "Qual è la capitale della Germania?", "La capitale della Germania è Berlino.", "geografia"),
    ("geografia-capitale-portogallo", "Qual è la capitale del Portogallo?", "La capitale del Portogallo è Lisbona.", "geografia"),
    ("geografia-capitale-giappone", "Qual è la capitale del Giappone?", "La capitale del Giappone è Tokyo.", "geografia"),
    ("geografia-capitale-canada", "Qual è la capitale del Canada?", "La capitale del Canada è Ottawa.", "geografia"),
    ("geografia-capitale-australia", "Qual è la capitale dell'Australia?", "La capitale dell'Australia è Canberra, non Sydney.", "geografia"),
    ("geografia-capitale-brasile", "Qual è la capitale del Brasile?", "La capitale del Brasile è Brasilia.", "geografia"),
    ("geografia-continenti", "Quanti sono i continenti secondo il modello scolastico più comune?", "Nel modello scolastico più comune i continenti sono sei: Europa, Asia, Africa, America, Oceania e Antartide.", "geografia"),
    ("geografia-oceano-grande", "Qual è l'oceano più grande della Terra?", "L'oceano più grande della Terra è l'Oceano Pacifico.", "geografia"),
    ("geografia-monte-alto", "Qual è la montagna più alta della Terra sul livello del mare?", "La montagna più alta sul livello del mare è l'Everest.", "geografia"),
    ("geografia-fiume-italiano", "Qual è il fiume più lungo d'Italia?", "Il fiume più lungo d'Italia è il Po.", "geografia"),
    ("geografia-stretto-sicilia", "Quale stretto separa la Sicilia dalla penisola italiana?", "Lo Stretto di Messina separa la Sicilia dalla Calabria.", "geografia"),
    ("scienza-acqua", "Qual è la formula chimica dell'acqua?", "La formula chimica dell'acqua è H₂O: due atomi di idrogeno e uno di ossigeno.", "scienza"),
    ("scienza-ossigeno", "Qual è il simbolo chimico dell'ossigeno?", "Il simbolo chimico dell'ossigeno è O.", "scienza"),
    ("scienza-oro", "Qual è il simbolo chimico dell'oro?", "Il simbolo chimico dell'oro è Au.", "scienza"),
    ("scienza-pianeta-rosso", "Quale pianeta è chiamato Pianeta Rosso?", "Il Pianeta Rosso è Marte.", "scienza"),
    ("scienza-pianeta-anelli", "Quale pianeta è famoso per i suoi anelli?", "Saturno è famoso per il suo sistema di anelli.", "scienza"),
    ("scienza-pianeta-vicino-sole", "Qual è il pianeta più vicino al Sole?", "Il pianeta più vicino al Sole è Mercurio.", "scienza"),
    ("scienza-sole-stella", "Il Sole è un pianeta o una stella?", "Il Sole è una stella: produce energia attraverso reazioni di fusione nucleare.", "scienza"),
    ("scienza-fotosintesi", "Che cosa produce la fotosintesi clorofilliana?", "La fotosintesi usa luce, acqua e anidride carbonica per produrre zuccheri e liberare ossigeno.", "scienza"),
    ("scienza-stati-materia", "Quali sono i tre stati classici della materia?", "I tre stati classici della materia sono solido, liquido e aeriforme o gassoso.", "scienza"),
    ("scienza-ebollizione", "A quale temperatura bolle l'acqua a livello del mare?", "A livello del mare l'acqua bolle circa a 100 gradi Celsius.", "scienza"),
    ("scienza-gravita", "Che cosa fa la gravità terrestre?", "La gravità attira gli oggetti verso il centro della Terra e ci mantiene sulla sua superficie.", "scienza"),
    ("informatica-ram", "Che differenza c'è tra RAM e spazio di archiviazione?", "La RAM conserva temporaneamente i dati usati dai programmi aperti. Lo spazio di archiviazione conserva file e app anche quando il dispositivo è spento.", "informatica"),
    ("informatica-browser", "Che cos'è un browser?", "Un browser è un programma usato per visitare siti e servizi sul web, per esempio Safari, Firefox o Chrome.", "informatica"),
    ("informatica-sistema-operativo", "Che cos'è un sistema operativo?", "Il sistema operativo è il software di base che gestisce hardware, file e programmi di un computer o telefono.", "informatica"),
    ("informatica-backup", "Che cos'è un backup?", "Un backup è una copia di sicurezza dei dati, utile per recuperarli dopo perdita, guasto o errore.", "informatica"),
    ("informatica-password", "Perché una password dovrebbe essere lunga e unica?", "Una password lunga e unica è più difficile da indovinare e limita i danni se un altro account viene compromesso.", "informatica"),
    ("matematica-percentuale", "Quanto fa il 10 per cento di 250?", "Il 10 per cento di 250 è 25.", "matematica"),
    ("matematica-frazione", "Quanto fa un quarto di 100?", "Un quarto di 100 è 25.", "matematica"),
    ("matematica-perimetro", "Come si calcola il perimetro di un rettangolo?", "Si sommano tutti i lati: per un rettangolo è due volte la base più due volte l'altezza.", "matematica"),
    ("matematica-area-rettangolo", "Come si calcola l'area di un rettangolo?", "L'area di un rettangolo si calcola moltiplicando base per altezza.", "matematica"),
    ("matematica-media", "Come si calcola la media aritmetica?", "Si sommano tutti i valori e si divide il risultato per il numero dei valori.", "matematica"),
    ("storia-unita-italia", "In quale anno fu proclamato il Regno d'Italia?", "Il Regno d'Italia fu proclamato nel 1861.", "storia"),
    ("storia-costituzione", "Quando entrò in vigore la Costituzione italiana?", "La Costituzione italiana entrò in vigore il 1 gennaio 1948.", "storia"),
    ("storia-muro-berlino", "In quale anno cadde il Muro di Berlino?", "Il Muro di Berlino cadde nel 1989.", "storia"),
    ("storia-rinascimento", "In quale paese ebbe origine il Rinascimento?", "Il Rinascimento ebbe origine in Italia e si sviluppò poi in altre parti d'Europa.", "storia"),
    ("letteratura-divina-commedia", "Chi ha scritto la Divina Commedia?", "La Divina Commedia è stata scritta da Dante Alighieri.", "letteratura"),
    ("letteratura-promessi-sposi", "Chi ha scritto I promessi sposi?", "I promessi sposi è un romanzo di Alessandro Manzoni.", "letteratura"),
    ("arte-gioconda", "Chi ha dipinto la Gioconda?", "La Gioconda è attribuita a Leonardo da Vinci.", "arte"),
    ("arte-guernica", "Chi ha dipinto Guernica?", "Guernica è un dipinto di Pablo Picasso.", "arte"),
    ("lingua-nome-aggettivo", "Qual è la differenza tra nome e aggettivo?", "Il nome indica persone, animali, cose o idee; l'aggettivo aggiunge una qualità o una caratteristica al nome.", "lingua"),
    ("lingua-sinonimo", "Che cos'è un sinonimo?", "Un sinonimo è una parola con significato uguale o molto simile a quello di un'altra parola.", "lingua"),
    ("lingua-verbo", "Che cos'è un verbo?", "Un verbo è una parola che esprime un'azione, uno stato o un evento.", "lingua"),
    ("educazione-civica-democrazia", "Che cos'è una democrazia?", "Una democrazia è una forma di governo in cui i cittadini partecipano alle decisioni, direttamente o attraverso rappresentanti eletti.", "educazione_civica"),
    ("educazione-civica-costituzione", "Che cos'è una Costituzione?", "Una Costituzione è la legge fondamentale di uno Stato: definisce diritti, doveri e funzionamento delle istituzioni.", "educazione_civica"),
    ("vita-quotidiana-frigorifero", "A che cosa serve il frigorifero?", "Il frigorifero mantiene gli alimenti a bassa temperatura per rallentare il deterioramento.", "vita_quotidiana"),
    ("vita-quotidiana-raccolta-differenziata", "Che cosa significa raccolta differenziata?", "La raccolta differenziata separa i rifiuti per materiale, così una parte può essere recuperata o riciclata correttamente.", "vita_quotidiana"),
]

CURRENT_FACTS = [
    {
        "id": "carica-presidente-repubblica-italiana",
        "question": "Chi è il Presidente della Repubblica Italiana?",
        "answer": "Alla data del 27 agosto 2026, il Presidente della Repubblica Italiana è Sergio Mattarella.",
        "source": "Presidenza della Repubblica",
        "url": "https://www.quirinale.it/allegati_statici/biografia/SergioMattarella.pdf",
    },
    {
        "id": "carica-presidente-consiglio-italiano",
        "question": "Chi è il Presidente del Consiglio dei ministri italiano?",
        "answer": "Alla data del 27 agosto 2026, il Presidente del Consiglio dei ministri italiano è Giorgia Meloni.",
        "source": "Presidenza del Consiglio dei Ministri",
        "url": "https://presidenza.governo.it/AmministrazioneTrasparente/Organizzazione/OrganiIndirizzoPolitico/GovernoMeloni/Pres_Meloni.html",
    },
    {
        "id": "carica-presidente-commissione-europea",
        "question": "Chi presiede la Commissione europea?",
        "answer": "Alla data del 27 agosto 2026, la Presidente della Commissione europea è Ursula von der Leyen.",
        "source": "Commissione europea",
        "url": "https://commission.europa.eu/about/organisation/president_en",
    },
    {
        "id": "carica-segretario-generale-onu",
        "question": "Chi è il Segretario generale delle Nazioni Unite?",
        "answer": "Alla data del 27 agosto 2026, il Segretario generale delle Nazioni Unite è António Guterres.",
        "source": "Nazioni Unite",
        "url": "https://www.un.org/en/about-us",
    },
]


def record(record_id: str, question: str, answer: str, category: str, provenance: dict) -> dict:
    return {
        "id": f"local-curated:factual-qa-v1:{record_id}",
        "source": "local-curated:factual-qa-v1",
        "license": "USER_APPROVED",
        "approval_status": "approved",
        "category": category,
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        "provenance": provenance,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--include-current",
        action="store_true",
        help="include time-sensitive office-holder snapshots dated 2026-08-27",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")

    rows = [
        record(
            fact_id, question, answer, category,
            {"kind": "evergreen_fact", "generator": "utils/sft/generate_italiano_factual_qa_v1.py"},
        )
        for fact_id, question, answer, category in EVERGREEN_FACTS
    ]
    if args.include_current:
        rows.extend(
            record(
                fact["id"], fact["question"], fact["answer"], "attualita_snapshot",
                {
                    "kind": "time_sensitive_snapshot",
                    "valid_at": "2026-08-27",
                    "source_name": fact["source"],
                    "source_url": fact["url"],
                    "generator": "utils/sft/generate_italiano_factual_qa_v1.py",
                },
            )
            for fact in CURRENT_FACTS
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as destination:
        for row in rows:
            destination.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps({"output": str(args.output), "examples": len(rows), "current_snapshot_included": args.include_current}))


if __name__ == "__main__":
    main()
