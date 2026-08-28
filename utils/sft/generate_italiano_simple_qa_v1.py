#!/usr/bin/env python3
"""Generate a controlled Italian simple-QA corpus for a 75M model.

The corpus intentionally targets narrow, measurable capabilities: stable
general knowledge, elementary arithmetic, short definitions, and simple text
transformations.  Facts are split as groups, so paraphrases of one fact never
leak across train, validation, and test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


SOURCE = "local-curated:italiano-simple-qa-v1"
SCRIPT = "utils/sft/generate_italiano_simple_qa_v1.py"
SPACE = re.compile(r"\s+")


CAPITALS = [
    ("Italia", "Roma"), ("Francia", "Parigi"), ("Spagna", "Madrid"),
    ("Germania", "Berlino"), ("Portogallo", "Lisbona"), ("Austria", "Vienna"),
    ("Belgio", "Bruxelles"), ("Paesi Bassi", "Amsterdam"), ("Danimarca", "Copenaghen"),
    ("Norvegia", "Oslo"), ("Svezia", "Stoccolma"), ("Finlandia", "Helsinki"),
    ("Islanda", "Reykjavík"), ("Irlanda", "Dublino"), ("Regno Unito", "Londra"),
    ("Polonia", "Varsavia"), ("Repubblica Ceca", "Praga"), ("Slovacchia", "Bratislava"),
    ("Ungheria", "Budapest"), ("Romania", "Bucarest"), ("Bulgaria", "Sofia"),
    ("Grecia", "Atene"), ("Croazia", "Zagabria"), ("Slovenia", "Lubiana"),
    ("Serbia", "Belgrado"), ("Albania", "Tirana"), ("Turchia", "Ankara"),
    ("Ucraina", "Kyiv"), ("Canada", "Ottawa"), ("Stati Uniti", "Washington D.C."),
    ("Messico", "Città del Messico"), ("Brasile", "Brasilia"),
    ("Argentina", "Buenos Aires"), ("Cile", "Santiago"), ("Perù", "Lima"),
    ("Colombia", "Bogotá"), ("Marocco", "Rabat"), ("Algeria", "Algeri"),
    ("Tunisia", "Tunisi"), ("Egitto", "Il Cairo"), ("Senegal", "Dakar"),
    ("Kenya", "Nairobi"), ("India", "Nuova Delhi"), ("Cina", "Pechino"),
    ("Giappone", "Tokyo"), ("Corea del Sud", "Seul"), ("Vietnam", "Hanoi"),
    ("Australia", "Canberra"), ("Nuova Zelanda", "Wellington"),
]

CONTINENTS = [
    ("Italia", "Europa"), ("Portogallo", "Europa"), ("Norvegia", "Europa"),
    ("Egitto", "Africa"), ("Kenya", "Africa"), ("Marocco", "Africa"),
    ("Senegal", "Africa"), ("India", "Asia"), ("Cina", "Asia"),
    ("Giappone", "Asia"), ("Vietnam", "Asia"), ("Canada", "America settentrionale"),
    ("Messico", "America settentrionale"), ("Brasile", "America meridionale"),
    ("Argentina", "America meridionale"), ("Cile", "America meridionale"),
    ("Perù", "America meridionale"), ("Australia", "Oceania"),
    ("Nuova Zelanda", "Oceania"),
]

ELEMENTS = [
    ("idrogeno", "H"), ("elio", "He"), ("carbonio", "C"), ("azoto", "N"),
    ("ossigeno", "O"), ("sodio", "Na"), ("magnesio", "Mg"), ("alluminio", "Al"),
    ("silicio", "Si"), ("fosforo", "P"), ("zolfo", "S"), ("cloro", "Cl"),
    ("potassio", "K"), ("calcio", "Ca"), ("ferro", "Fe"), ("rame", "Cu"),
    ("argento", "Ag"), ("oro", "Au"), ("mercurio", "Hg"), ("piombo", "Pb"),
]

SCIENCE_FACTS = [
    ("formula-acqua", "Qual è la formula chimica dell'acqua?", "La formula chimica dell'acqua è H₂O."),
    ("pianeta-rosso", "Quale pianeta è chiamato Pianeta Rosso?", "Il Pianeta Rosso è Marte."),
    ("pianeta-vicino-sole", "Qual è il pianeta più vicino al Sole?", "Il pianeta più vicino al Sole è Mercurio."),
    ("pianeta-grande", "Qual è il pianeta più grande del Sistema Solare?", "Il pianeta più grande del Sistema Solare è Giove."),
    ("anelli", "Quale pianeta è famoso per i suoi anelli?", "Saturno è famoso per il suo sistema di anelli."),
    ("satellite-terra", "Come si chiama il satellite naturale della Terra?", "Il satellite naturale della Terra è la Luna."),
    ("sole", "Il Sole è un pianeta o una stella?", "Il Sole è una stella."),
    ("ebollizione", "A quale temperatura bolle l'acqua a livello del mare?", "A livello del mare l'acqua bolle circa a 100 gradi Celsius."),
    ("congelamento", "A quale temperatura congela l'acqua pura?", "L'acqua pura congela circa a 0 gradi Celsius."),
    ("stati-materia", "Quali sono i tre stati classici della materia?", "I tre stati classici della materia sono solido, liquido e gassoso."),
    ("gravita", "Che cosa fa la gravità terrestre?", "La gravità attira gli oggetti verso il centro della Terra."),
    ("fotosintesi", "Che cosa fanno le piante durante la fotosintesi?", "Usano luce, acqua e anidride carbonica per produrre zuccheri e liberare ossigeno."),
    ("cuore", "Qual è la funzione principale del cuore?", "Il cuore pompa il sangue attraverso il corpo."),
    ("polmoni", "A che cosa servono i polmoni?", "I polmoni permettono gli scambi di ossigeno e anidride carbonica durante la respirazione."),
    ("scheletro", "A che cosa serve lo scheletro umano?", "Lo scheletro sostiene il corpo, protegge organi e collabora al movimento."),
    ("vertebrati", "Che cosa distingue un vertebrato?", "Un vertebrato possiede una colonna vertebrale."),
    ("mammiferi", "Quale caratteristica distingue i mammiferi?", "I mammiferi nutrono i piccoli con il latte prodotto dalle ghiandole mammarie."),
    ("api", "Che ruolo hanno le api nell'impollinazione?", "Trasportano polline tra i fiori e favoriscono la riproduzione di molte piante."),
    ("velocita-luce", "La luce è più veloce del suono?", "Sì. La luce viaggia molto più velocemente del suono."),
    ("termometro", "Che cosa misura un termometro?", "Un termometro misura la temperatura."),
    ("barometro", "Che cosa misura un barometro?", "Un barometro misura la pressione atmosferica."),
    ("microscopio", "A che cosa serve un microscopio?", "Un microscopio permette di osservare oggetti troppo piccoli per essere visti a occhio nudo."),
    ("energia-rinnovabile", "Che cos'è una fonte di energia rinnovabile?", "È una fonte che si rigenera naturalmente, come sole, vento o acqua."),
    ("ecosistema", "Che cos'è un ecosistema?", "È l'insieme degli organismi e dell'ambiente con cui interagiscono."),
    ("atomo", "Che cos'è un atomo?", "Un atomo è una delle unità fondamentali che compongono la materia."),
]

HUMANITIES_FACTS = [
    ("regno-italia", "In quale anno fu proclamato il Regno d'Italia?", "Il Regno d'Italia fu proclamato nel 1861."),
    ("costituzione", "Quando entrò in vigore la Costituzione italiana?", "La Costituzione italiana entrò in vigore il 1 gennaio 1948."),
    ("muro-berlino", "In quale anno cadde il Muro di Berlino?", "Il Muro di Berlino cadde nel 1989."),
    ("rinascimento", "In quale paese ebbe origine il Rinascimento?", "Il Rinascimento ebbe origine in Italia."),
    ("romani-lingua", "Quale lingua parlavano gli antichi Romani?", "Gli antichi Romani parlavano principalmente latino."),
    ("dante", "Chi ha scritto la Divina Commedia?", "La Divina Commedia è stata scritta da Dante Alighieri."),
    ("manzoni", "Chi ha scritto I promessi sposi?", "I promessi sposi è un romanzo di Alessandro Manzoni."),
    ("pinocchio", "Chi ha scritto Le avventure di Pinocchio?", "Le avventure di Pinocchio è stato scritto da Carlo Collodi."),
    ("decameron", "Chi ha scritto il Decameron?", "Il Decameron è stato scritto da Giovanni Boccaccio."),
    ("orlando-furioso", "Chi ha scritto l'Orlando furioso?", "L'Orlando furioso è stato scritto da Ludovico Ariosto."),
    ("gioconda", "Chi ha dipinto la Gioconda?", "La Gioconda è attribuita a Leonardo da Vinci."),
    ("guernica", "Chi ha dipinto Guernica?", "Guernica è un dipinto di Pablo Picasso."),
    ("notte-stellata", "Chi ha dipinto La notte stellata?", "La notte stellata è un dipinto di Vincent van Gogh."),
    ("david", "Chi ha scolpito il David conservato alla Galleria dell'Accademia di Firenze?", "Il David è stato scolpito da Michelangelo Buonarroti."),
    ("venere-botticelli", "Chi ha dipinto La nascita di Venere?", "La nascita di Venere è un dipinto di Sandro Botticelli."),
    ("mozart", "In quale paese nacque Wolfgang Amadeus Mozart?", "Mozart nacque a Salisburgo, nell'attuale Austria."),
    ("beethoven", "Beethoven era un compositore o un pittore?", "Ludwig van Beethoven era un compositore."),
    ("colosseo", "In quale città si trova il Colosseo?", "Il Colosseo si trova a Roma."),
    ("torre-pisa", "In quale città si trova la celebre Torre pendente?", "La celebre Torre pendente si trova a Pisa."),
    ("uffizi", "In quale città si trova la Galleria degli Uffizi?", "La Galleria degli Uffizi si trova a Firenze."),
]

DEFINITIONS = [
    ("browser", "un browser", "Un browser è un programma usato per visitare siti e servizi sul web."),
    ("sistema-operativo", "un sistema operativo", "Un sistema operativo è il software di base che gestisce hardware, file e programmi."),
    ("backup", "un backup", "Un backup è una copia di sicurezza dei dati."),
    ("ram", "la memoria RAM", "La RAM conserva temporaneamente i dati usati dai programmi aperti."),
    ("archiviazione", "lo spazio di archiviazione", "Lo spazio di archiviazione conserva file e applicazioni anche a dispositivo spento."),
    ("file", "un file", "Un file è un insieme di dati salvato con un nome su un dispositivo."),
    ("cartella", "una cartella informatica", "Una cartella serve a organizzare file e altre cartelle."),
    ("rete", "una rete informatica", "Una rete informatica collega dispositivi affinché possano comunicare e condividere risorse."),
    ("internet", "Internet", "Internet è una rete mondiale di reti informatiche collegate tra loro."),
    ("web", "il World Wide Web", "Il Web è un insieme di pagine e risorse accessibili tramite Internet."),
    ("email", "un'email", "Un'email è un messaggio inviato attraverso la posta elettronica."),
    ("password", "una password", "Una password è una sequenza segreta usata per proteggere l'accesso a un account."),
    ("autenticazione-due-fattori", "l'autenticazione a due fattori", "Aggiunge una seconda verifica oltre alla password."),
    ("wifi", "il Wi-Fi", "Il Wi-Fi permette ai dispositivi di collegarsi a una rete senza usare cavi di rete."),
    ("bluetooth", "il Bluetooth", "Bluetooth collega dispositivi vicini tramite una comunicazione radio a corto raggio."),
    ("app", "un'applicazione", "Un'applicazione è un programma progettato per svolgere funzioni specifiche."),
    ("database", "un database", "Un database è una raccolta organizzata di dati che può essere consultata e aggiornata."),
    ("algoritmo", "un algoritmo", "Un algoritmo è una sequenza ordinata di istruzioni per risolvere un problema."),
    ("cpu", "la CPU", "La CPU esegue istruzioni e coordina molte operazioni del computer."),
    ("cloud", "il cloud computing", "Il cloud computing fornisce risorse informatiche attraverso una rete, spesso Internet."),
]

LANGUAGE_FACTS = [
    ("nome", "Che cos'è un nome in grammatica?", "Il nome indica una persona, un animale, una cosa, un luogo o un'idea."),
    ("aggettivo", "Che cos'è un aggettivo?", "L'aggettivo aggiunge una qualità o una caratteristica a un nome."),
    ("verbo", "Che cos'è un verbo?", "Il verbo esprime un'azione, uno stato o un evento."),
    ("avverbio", "Che cos'è un avverbio?", "L'avverbio modifica o precisa il significato di un verbo, un aggettivo o un altro avverbio."),
    ("pronome", "Che cos'è un pronome?", "Il pronome sostituisce un nome o un gruppo nominale."),
    ("sinonimo", "Che cos'è un sinonimo?", "Un sinonimo è una parola con significato uguale o molto simile a un'altra."),
    ("contrario", "Che cos'è un contrario?", "Un contrario è una parola con significato opposto a un'altra."),
    ("soggetto", "Che cos'è il soggetto di una frase?", "Il soggetto è la persona, l'animale o la cosa di cui si parla o che compie l'azione."),
    ("frase", "Che cos'è una frase?", "Una frase è un insieme organizzato di parole che comunica un significato completo."),
    ("paragrafo", "Che cos'è un paragrafo?", "Un paragrafo è una parte di testo formata da frasi dedicate a uno stesso argomento."),
    ("plurale-casa", "Qual è il plurale di casa?", "Il plurale di casa è case."),
    ("plurale-uomo", "Qual è il plurale di uomo?", "Il plurale di uomo è uomini."),
    ("plurale-ciliegia", "Qual è il plurale di ciliegia?", "Il plurale di ciliegia è ciliegie."),
    ("passato-mangiare", "Qual è il participio passato di mangiare?", "Il participio passato di mangiare è mangiato."),
    ("passato-scrivere", "Qual è il participio passato di scrivere?", "Il participio passato di scrivere è scritto."),
]

PRACTICAL = [
    ("email-informazioni", "Come posso iniziare un'email per chiedere informazioni?", "Puoi iniziare con: «Buongiorno, vorrei chiedere alcune informazioni riguardo al servizio»."),
    ("email-chiusura", "Come posso chiudere un'email professionale?", "Puoi scrivere: «Grazie per l'attenzione. Cordiali saluti»."),
    ("messaggio-gentile", "Rendi più gentile: Mandami il documento oggi.", "Potresti inviarmi il documento entro oggi, per favore?"),
    ("ritardo", "Come avviso un collega che arriverò in ritardo?", "Puoi scrivere: «Ciao, arriverò con circa dieci minuti di ritardo. Mi scuso per l'attesa»."),
    ("lista-spesa", "Come posso organizzare una lista della spesa?", "Dividila per reparti, per esempio frutta, dispensa, frigorifero e prodotti per la casa."),
    ("studio-breve", "Come organizzo un'ora di studio?", "Dedica 40 minuti allo studio, 10 al ripasso senza appunti e 10 alla verifica degli errori."),
    ("scrivania", "Come posso riordinare velocemente la scrivania?", "Elimina ciò che non serve, raggruppa gli oggetti simili e lascia a portata di mano solo quelli usati spesso."),
    ("file", "Come posso organizzare i file di un progetto?", "Crea una cartella principale e sottocartelle chiare per documenti, risorse, bozze e versioni finali."),
    ("password", "Come creo una password più sicura?", "Usa una frase lunga e unica e attiva l'autenticazione a due fattori quando disponibile."),
    ("backup", "Come posso proteggere le mie foto da una perdita accidentale?", "Conserva almeno una seconda copia su un disco esterno o su un servizio affidabile e verifica periodicamente il backup."),
    ("presentazione", "Come strutturo una presentazione di cinque minuti?", "Apri con il messaggio principale, sviluppa due o tre punti e chiudi con un breve riepilogo."),
    ("budget", "Da dove comincio per creare un budget mensile?", "Annota entrate, spese fisse e spese variabili, poi stabilisci una piccola quota per risparmio e imprevisti."),
    ("concentrazione", "Che cosa posso fare se mi distraggo mentre studio?", "Allontana le distrazioni, scegli un compito preciso e lavora per un intervallo breve prima di fare una pausa."),
    ("riassunto", "Come posso scrivere un buon riassunto?", "Individua le idee principali, elimina esempi secondari e riscrivi i concetti con parole tue."),
    ("istruzioni", "Come posso rendere più chiare delle istruzioni?", "Usa passaggi brevi in ordine, un'azione per punto e indica chiaramente il risultato atteso."),
]


def normalise(text: str) -> str:
    return SPACE.sub(" ", text).strip().casefold()


def fnv1a_64(text: str) -> int:
    value = 14695981039346656037
    for byte in text.encode("utf-8"):
        value ^= byte
        value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return value


def group_split(group_key: str) -> str:
    bucket = int.from_bytes(hashlib.sha256(group_key.encode("utf-8")).digest()[:8], "big") % 10_000
    return "train" if bucket < 9000 else "validation" if bucket < 9500 else "test"


def id_for_split(group_key: str, variant: int, target: str) -> str:
    for nonce in range(100_000):
        candidate = f"{SOURCE}:{group_key}:v{variant:02d}:{nonce}"
        bucket = fnv1a_64(candidate) % 10_000
        actual = "train" if bucket < 9000 else "validation" if bucket < 9500 else "test"
        if actual == target:
            return candidate
    raise RuntimeError(f"could not assign {group_key} variant {variant} to {target}")


def record(group_key: str, category: str, variant: int, question: str, answer: str, split: str) -> dict:
    return {
        "id": id_for_split(group_key, variant, split),
        "source": SOURCE,
        "license": "USER_APPROVED",
        "approval_status": "approved",
        "category": category,
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        "provenance": {
            "kind": "controlled_synthetic_simple_qa",
            "generator": SCRIPT,
            "fact_group": group_key,
            "intended_split": split,
            "variant": variant,
        },
    }


def add_group(rows: list[dict], evaluations: list[dict], group_key: str, category: str,
              questions: list[str], answers: list[str], eval_question: str | None = None) -> None:
    split = group_split(group_key)
    for index, question in enumerate(questions, start=1):
        rows.append(record(group_key, category, index, question, answers[(index - 1) % len(answers)], split))
    if split == "train" and eval_question is not None:
        evaluations.append({
            "id": f"manual-eval:{group_key}",
            "category": category,
            "fact_group": group_key,
            "prompt": eval_question,
            "expected_answer": answers[0],
            "trained_prompt_variants": len(questions),
        })


def add_capitals(rows: list[dict], evaluations: list[dict]) -> None:
    for country, capital in CAPITALS:
        key = f"capitale-{normalise(country).replace(' ', '-')}"
        if country == "Italia":
            country_with_of = "d'Italia"
        elif country == "Paesi Bassi":
            country_with_of = "dei Paesi Bassi"
        elif country == "Stati Uniti":
            country_with_of = "degli Stati Uniti"
        elif country[0].lower() in "aeiou":
            country_with_of = f"dell'{country}"
        elif country in {"Portogallo", "Belgio", "Regno Unito", "Canada", "Messico",
                         "Brasile", "Marocco", "Senegal", "Kenya", "Giappone", "Vietnam"}:
            country_with_of = f"del {country}"
        else:
            country_with_of = f"della {country}"
        add_group(rows, evaluations, key, "geografia",
                  [f"Qual è la capitale {country_with_of}?", f"Come si chiama la capitale {country_with_of}?",
                   f"Dimmi la capitale {country_with_of}.", f"{country}: qual è la sua capitale?"],
                  [f"La capitale {country_with_of} è {capital}.", f"È {capital}.",
                   f"{capital} è la capitale {country_with_of}."],
                  f"Sai indicarmi qual è la capitale {country_with_of}?")


def add_continents(rows: list[dict], evaluations: list[dict]) -> None:
    for country, continent in CONTINENTS:
        key = f"continente-{normalise(country).replace(' ', '-')}"
        add_group(rows, evaluations, key, "geografia",
                  [f"In quale continente si trova {country}?", f"A quale continente appartiene {country}?",
                   f"{country} si trova in quale continente?"],
                  [f"{country} si trova in {continent}.", f"Il continente è {continent}."],
                  f"Dove si colloca {country} a livello di continente?")


def add_elements(rows: list[dict], evaluations: list[dict]) -> None:
    for name, symbol in ELEMENTS:
        key = f"simbolo-{name}"
        element_with_of = f"dell'{name}" if name[0].lower() in "aeiou" else f"del {name}"
        element_with_article = (
            f"l'{name}" if name[0].lower() in "aeiou" else f"lo {name}" if name == "zolfo" else f"il {name}"
        )
        add_group(rows, evaluations, key, "scienza",
                  [f"Qual è il simbolo chimico {element_with_of}?", f"Con quale simbolo si indica {element_with_article}?",
                   f"Dimmi il simbolo dell'elemento {name}."],
                  [f"Il simbolo chimico {element_with_of} è {symbol}.", f"Il simbolo è {symbol}."],
                  f"Come si abbrevia l'elemento {name} nella tavola periodica?")


def paraphrased_fact(rows: list[dict], evaluations: list[dict], category: str,
                     key: str, question: str, answer: str) -> None:
    stem = question[:-1] if question.endswith("?") else question
    add_group(rows, evaluations, key, category,
              [question, f"Rispondi in breve: {question}", f"Fornisci una risposta semplice: {question}"],
              [answer], f"Mi sai dire questo: {stem.lower()}?")


def add_definitions(rows: list[dict], evaluations: list[dict]) -> None:
    for key, term, answer in DEFINITIONS:
        add_group(rows, evaluations, f"informatica-{key}", "informatica",
                  [f"Che cos'è {term}?", f"Spiega in breve che cos'è {term}.",
                   f"Definisci {term} in modo semplice."], [answer],
                  f"Come spiegheresti {term} a un principiante?")


def add_arithmetic(rows: list[dict], evaluations: list[dict]) -> None:
    additions = [(7, 5), (12, 9), (15, 18), (24, 16), (31, 27), (45, 14), (56, 23),
                 (64, 28), (73, 19), (82, 17), (105, 36), (120, 75), (250, 125),
                 (99, 101), (340, 60), (17, 44), (29, 38), (66, 33), (48, 52), (135, 65)]
    subtractions = [(12, 5), (20, 7), (31, 16), (45, 18), (60, 25), (73, 29),
                    (90, 34), (100, 48), (125, 25), (200, 75), (350, 120), (81, 19),
                    (64, 32), (150, 55), (99, 40)]
    multiplications = [(2, 7), (3, 9), (4, 8), (5, 12), (6, 7), (7, 8), (8, 9),
                       (9, 9), (11, 6), (12, 4), (12, 8), (15, 3), (20, 5), (25, 4),
                       (14, 6), (16, 5), (18, 4), (21, 3), (30, 7), (40, 6)]
    percentages = [(10, 50), (10, 80), (10, 250), (20, 50), (20, 150), (25, 40),
                   (25, 80), (25, 200), (50, 30), (50, 90), (50, 240), (5, 200),
                   (5, 400), (75, 40), (75, 120)]

    for a, b in additions:
        result = a + b
        key = f"somma-{a}-{b}"
        add_group(rows, evaluations, key, "matematica",
                  [f"Quanto fa {a} più {b}?", f"Calcola la somma di {a} e {b}."],
                  [f"{a} più {b} fa {result}.", f"La somma è {result}."],
                  f"Qual è il risultato di {a} + {b}?")
    for a, b in subtractions:
        result = a - b
        key = f"differenza-{a}-{b}"
        add_group(rows, evaluations, key, "matematica",
                  [f"Quanto fa {a} meno {b}?", f"Calcola la differenza tra {a} e {b}."],
                  [f"{a} meno {b} fa {result}.", f"La differenza è {result}."],
                  f"Qual è il risultato di {a} - {b}?")
    for a, b in multiplications:
        result = a * b
        key = f"prodotto-{a}-{b}"
        add_group(rows, evaluations, key, "matematica",
                  [f"Quanto fa {a} per {b}?", f"Calcola il prodotto di {a} e {b}."],
                  [f"{a} per {b} fa {result}.", f"Il prodotto è {result}."],
                  f"Qual è il risultato di {a} × {b}?")
    for percentage, value in percentages:
        result = percentage * value // 100
        key = f"percentuale-{percentage}-{value}"
        add_group(rows, evaluations, key, "matematica",
                  [f"Quanto fa il {percentage} per cento di {value}?",
                   f"Calcola il {percentage}% di {value}."],
                  [f"Il {percentage} per cento di {value} è {result}.", f"Il risultato è {result}."],
                  f"A quanto corrisponde il {percentage}% di {value}?")


def build() -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    evaluations: list[dict] = []
    add_capitals(rows, evaluations)
    add_continents(rows, evaluations)
    add_elements(rows, evaluations)
    for key, question, answer in SCIENCE_FACTS:
        paraphrased_fact(rows, evaluations, "scienza", f"scienza-{key}", question, answer)
    for key, question, answer in HUMANITIES_FACTS:
        paraphrased_fact(rows, evaluations, "cultura_umanistica", f"umanistica-{key}", question, answer)
    add_definitions(rows, evaluations)
    for key, question, answer in LANGUAGE_FACTS:
        paraphrased_fact(rows, evaluations, "lingua_italiana", f"lingua-{key}", question, answer)
    for key, question, answer in PRACTICAL:
        add_group(rows, evaluations, f"pratico-{key}", "istruzioni_pratiche",
                  [question, f"Rispondi in modo pratico: {question}"], [answer],
                  f"Mi dai un consiglio semplice? {question}")
    add_arithmetic(rows, evaluations)
    return rows, evaluations


def validate(rows: list[dict], evaluations: list[dict]) -> dict:
    ids: set[str] = set()
    questions: set[str] = set()
    split_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    max_user = 0
    max_assistant = 0
    for row in rows:
        if row["id"] in ids:
            raise RuntimeError(f"duplicate id: {row['id']}")
        ids.add(row["id"])
        question = row["messages"][0]["content"]
        answer = row["messages"][1]["content"]
        normalised_question = normalise(question)
        if normalised_question in questions:
            raise RuntimeError(f"duplicate question: {question}")
        questions.add(normalised_question)
        if not question.strip() or not answer.strip():
            raise RuntimeError("empty question or answer")
        if len(question) > 220 or len(answer) > 320:
            raise RuntimeError(f"record too long: {row['id']}")
        split_counts[row["provenance"]["intended_split"]] += 1
        category_counts[row["category"]] += 1
        max_user = max(max_user, len(question))
        max_assistant = max(max_assistant, len(answer))
    evaluation_prompts = [normalise(item["prompt"]) for item in evaluations]
    if len(evaluation_prompts) != len(set(evaluation_prompts)):
        raise RuntimeError("duplicate manual evaluation prompt")
    if questions.intersection(evaluation_prompts):
        raise RuntimeError("manual evaluation prompt leaked into training source")
    return {
        "records": len(rows),
        "fact_groups": len({row["provenance"]["fact_group"] for row in rows}),
        "manual_evaluations": len(evaluations),
        "split_counts": dict(sorted(split_counts.items())),
        "category_counts": dict(sorted(category_counts.items())),
        "max_user_characters": max_user,
        "max_assistant_characters": max_assistant,
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8") as destination:
        for row in rows:
            destination.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    source_path = args.output_directory / "italiano-simple-qa-v1.source.jsonl"
    evaluation_path = args.output_directory / "italiano-simple-qa-v1.manual-eval.jsonl"
    report_path = args.output_directory / "italiano-simple-qa-v1.report.json"
    for path in (source_path, evaluation_path, report_path):
        if path.exists():
            raise SystemExit(f"refusing to overwrite existing output: {path}")

    rows, evaluations = build()
    report = {
        "schema": "italiano-simple-qa-v1-report",
        "source": SOURCE,
        "policy": {
            "current_affairs": "excluded",
            "medical_legal_financial": "excluded",
            "systems": "none",
            "maximum_user_characters": 220,
            "maximum_assistant_characters": 320,
            "split_unit": "fact_group",
            "manual_eval_in_training": False,
        },
        **validate(rows, evaluations),
    }
    args.output_directory.mkdir(parents=True, exist_ok=True)
    write_jsonl(source_path, rows)
    write_jsonl(evaluation_path, evaluations)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
