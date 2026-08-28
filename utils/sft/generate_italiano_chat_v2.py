#!/usr/bin/env python3
"""Generate a reproducible, multi-turn Italian SFT batch for internal use."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Task:
    category: str
    goal: str
    first_action: str
    second_action: str
    check: str


TASKS = [
    Task("organizzazione_casa", "riordinare una piccola dispensa", "svuota un solo ripiano e separa gli alimenti per tipo", "metti davanti quelli con scadenza più vicina", "controlla che ogni prodotto resti visibile"),
    Task("organizzazione_casa", "organizzare i cavi in un cassetto", "raccogli prima i cavi che usi davvero", "avvolgi ogni cavo senza stringerlo e applica un'etichetta", "verifica di poter riconoscere subito ogni estremità"),
    Task("organizzazione_casa", "preparare una cartellina per i documenti importanti", "scegli poche sezioni chiare, per esempio casa, lavoro e salute", "inserisci una copia digitale dei documenti più importanti", "controlla la cartellina una volta all'anno"),
    Task("organizzazione_casa", "tenere ordinato il piano della cucina", "lascia sul piano soltanto gli oggetti usati ogni giorno", "assegna un posto preciso agli utensili rimanenti", "verifica che la pulizia quotidiana richieda meno di cinque minuti"),
    Task("organizzazione_casa", "preparare una borsa per il giorno dopo", "metti insieme chiavi, portafoglio e gli oggetti indispensabili", "aggiungi ciò che dipende dall'impegno del giorno successivo", "controlla la lista prima di andare a dormire"),
    Task("organizzazione_casa", "gestire la posta cartacea", "apri la posta vicino a un cestino per il riciclo", "dividi subito tra azioni da fare, da conservare e da eliminare", "verifica di non lasciare documenti con scadenze senza promemoria"),
    Task("organizzazione_casa", "organizzare i prodotti per la pulizia", "raggruppa i prodotti per ambiente e tieni separati quelli pericolosi", "conserva le istruzioni e non travasare i liquidi in contenitori anonimi", "verifica che bambini e animali non possano raggiungerli"),
    Task("organizzazione_casa", "creare una lista della spesa riutilizzabile", "dividi la lista per reparti del supermercato", "aggiungi gli articoli quando stanno per finire", "controlla la dispensa prima di uscire"),
    Task("organizzazione_casa", "ridurre il disordine sul desktop del computer", "sposta i file importanti nelle cartelle Documenti o Progetti", "crea una cartella temporanea per ciò che deve ancora essere ordinato", "verifica che sul desktop restino solo collegamenti utili"),
    Task("organizzazione_casa", "organizzare una piccola biblioteca", "ordina prima i libri in grandi gruppi che riconosci facilmente", "riserva uno spazio ai libri da leggere e a quelli da prestare", "controlla che ogni gruppo abbia spazio per nuovi arrivi"),
    Task("studio", "preparare una sessione di studio di un'ora", "scegli un obiettivo osservabile, non soltanto una materia", "dividi l'ora in una fase di studio e una di richiamo senza appunti", "verifica alla fine che cosa sai spiegare a voce"),
    Task("studio", "ripassare un capitolo lungo", "scorri titoli, immagini e parole chiave prima di rileggere", "trasforma le sezioni in domande e prova a rispondere", "controlla i punti in cui esiti senza guardare il testo"),
    Task("studio", "creare flashcard utili", "scrivi una domanda precisa su un lato e una risposta breve sull'altro", "separa le carte difficili da quelle già note", "verifica che ogni carta richieda un solo concetto"),
    Task("studio", "preparare un esame orale", "elenca gli argomenti e scegli un ordine logico", "prova a esporre a voce registrandoti o con una persona", "controlla se sai collegare gli argomenti senza leggere"),
    Task("studio", "prendere appunti durante una lezione", "scrivi concetti e parole chiave invece di trascrivere tutto", "lascia margini per aggiungere esempi e dubbi", "rileggi gli appunti entro ventiquattro ore"),
    Task("studio", "gestire una distrazione durante lo studio", "annota il pensiero o l'attività che ti distrae su un foglio", "torna al compito per un intervallo breve e definito", "controlla se la distrazione è urgente solo alla pausa"),
    Task("studio", "imparare nuovi vocaboli in italiano", "scegli parole incontrate in un contesto reale", "scrivi una frase personale per ciascuna parola", "verifica di saperla usare senza rileggere la definizione"),
    Task("studio", "organizzare un progetto di gruppo", "definite un risultato concreto e una scadenza condivisa", "assegnate compiti con responsabile e data di controllo", "verificate insieme i progressi in incontri brevi"),
    Task("studio", "preparare una presentazione breve", "decidi prima il messaggio principale in una frase", "organizza poche slide che sostengano quel messaggio", "verifica che il discorso sia comprensibile anche senza leggere le slide"),
    Task("studio", "ricordare una formula matematica", "comprendi prima che cosa rappresentano le grandezze", "risolvi alcuni esercizi semplici applicandola", "verifica di saper riconoscere quando usarla"),
    Task("digitale", "mettere in sicurezza un nuovo computer", "installa gli aggiornamenti del sistema operativo", "attiva un blocco schermo e un sistema di backup", "verifica che gli account principali usino password diverse"),
    Task("digitale", "creare password più sicure", "usa frasi lunghe e uniche invece di parole brevi prevedibili", "affidati a un gestore di password se ne usi molti", "verifica di avere attivato l'autenticazione a due fattori dove disponibile"),
    Task("digitale", "liberare spazio sul telefono", "controlla prima foto, video e download più pesanti", "elimina o archivia ciò che non ti serve e disinstalla le app inutilizzate", "verifica lo spazio libero prima di aggiornare il sistema"),
    Task("digitale", "riconoscere un possibile messaggio di phishing", "osserva mittente, indirizzo web e richieste urgenti", "non aprire link o allegati se qualcosa sembra insolito", "verifica l'informazione entrando dal sito o dall'app ufficiale"),
    Task("digitale", "fare il backup delle foto", "scegli una seconda copia su disco esterno o servizio affidabile", "imposta una frequenza semplice e automatica quando possibile", "verifica aprendo qualche foto dalla copia di backup"),
    Task("digitale", "organizzare i file di un progetto", "crea una cartella principale con sottocartelle per bozze, risorse e versioni finali", "usa nomi di file con data o versione quando serve", "verifica che un'altra persona possa capire la struttura"),
    Task("digitale", "aggiornare un'app in modo prudente", "leggi le note essenziali e verifica di avere spazio libero", "esegui l'aggiornamento da uno store o sito ufficiale", "controlla subito che le funzioni importanti siano ancora disponibili"),
    Task("digitale", "condividere un documento senza perdere il controllo", "scegli il livello minimo di permesso necessario", "usa un link con scadenza se il servizio lo permette", "verifica chi può ancora accedere dopo la collaborazione"),
    Task("digitale", "risolvere un sito che non si carica", "controlla prima la connessione e prova a ricaricare", "apri il sito in una finestra privata o in un altro browser", "verifica se il problema riguarda solo quel sito"),
    Task("digitale", "scegliere quali notifiche lasciare attive", "mantieni quelle utili per persone e scadenze importanti", "disattiva le notifiche promozionali o ripetitive", "verifica per qualche giorno se ti manca davvero qualcosa"),
    Task("scrittura_comunicazione", "scrivere un'email per chiedere informazioni", "apri con il motivo della richiesta in una frase", "indica con chiarezza quali informazioni ti servono e entro quando", "verifica che destinatario e allegati siano corretti"),
    Task("scrittura_comunicazione", "avvisare un collega di un ritardo", "comunica subito il ritardo e una stima realistica dell'arrivo", "offri un'alternativa se l'attesa blocca il lavoro", "verifica di non dare una spiegazione più personale del necessario"),
    Task("scrittura_comunicazione", "rispondere a un reclamo in modo professionale", "riconosci il problema senza attribuire colpe premature", "spiega la prossima azione concreta e il tempo previsto", "verifica che il tono resti calmo e rispettoso"),
    Task("scrittura_comunicazione", "chiedere un feedback su un lavoro", "spiega su quale parte desideri un parere", "fai domande specifiche invece di chiedere un giudizio generico", "verifica di lasciare tempo sufficiente per rispondere"),
    Task("scrittura_comunicazione", "rifiutare un invito con gentilezza", "ringrazia per l'invito e comunica subito che non puoi partecipare", "se vuoi, proponi un'altra occasione senza promettere ciò che non puoi mantenere", "verifica che il messaggio sia breve e diretto"),
    Task("scrittura_comunicazione", "riassumere una riunione", "indica decisioni, responsabilità e prossimi passi", "separa i fatti dalle questioni ancora aperte", "verifica che ogni azione abbia un responsabile e una data"),
    Task("scrittura_comunicazione", "rendere più chiaro un messaggio lungo", "metti l'informazione più importante all'inizio", "trasforma i dettagli in punti brevi quando aiutano la lettura", "verifica che ogni frase aggiunga qualcosa di necessario"),
    Task("scrittura_comunicazione", "chiedere scusa per un errore", "riconosci l'errore senza giustificarlo troppo", "spiega come intendi rimediare e quando", "verifica di non scaricare la responsabilità su altre persone"),
    Task("scrittura_comunicazione", "preparare un messaggio di benvenuto", "saluta e spiega subito il contesto", "indica una o due informazioni utili per iniziare", "verifica che il messaggio non sia troppo lungo per il canale usato"),
    Task("scrittura_comunicazione", "fare una richiesta urgente senza essere brusco", "spiega perché la richiesta è urgente in modo concreto", "indica esattamente che cosa serve e entro quale orario", "verifica di lasciare spazio a un eventuale impedimento"),
    Task("vita_quotidiana", "preparare un pasto semplice nei giorni impegnati", "scegli una base come pasta, riso o legumi e pochi ingredienti", "prepara una quantità leggermente maggiore per il giorno dopo", "verifica di conservare gli avanzi in frigorifero in tempi brevi"),
    Task("vita_quotidiana", "gestire il bucato senza accumularlo", "separa i capi secondo colore e istruzioni di lavaggio", "programma lavaggi piccoli ma regolari", "verifica le tasche e le etichette prima di avviare la lavatrice"),
    Task("vita_quotidiana", "preparare una valigia leggera", "scegli capi che si combinano tra loro", "controlla meteo, durata e attività previste", "verifica di lasciare spazio per documenti e farmaci personali"),
    Task("vita_quotidiana", "innaffiare una pianta da appartamento", "controlla prima se il terriccio è asciutto in superficie", "versa acqua lentamente fino a inumidire il terriccio senza ristagni", "verifica che il vaso abbia un buon drenaggio"),
    Task("vita_quotidiana", "organizzare le spese mensili", "annota prima entrate e spese fisse", "stabilisci una piccola quota per imprevisti", "verifica a fine mese dove il piano è stato rispettato"),
    Task("vita_quotidiana", "pulire il frigorifero in modo ordinato", "svuota un ripiano alla volta e controlla le scadenze", "lava con un panno umido e asciuga prima di rimettere gli alimenti", "verifica che carne e pesce siano conservati nei contenitori adatti"),
    Task("vita_quotidiana", "ridurre gli sprechi alimentari", "pianifica i pasti partendo da ciò che hai già", "congela o riutilizza gli ingredienti vicini alla scadenza", "verifica una volta alla settimana ciò che rischia di non essere usato"),
    Task("vita_quotidiana", "organizzare un piccolo balcone", "misura lo spazio e osserva quante ore di luce riceve", "scegli pochi vasi adatti e lascia passaggi liberi", "verifica che l'acqua non ristagni sui pavimenti"),
    Task("vita_quotidiana", "preparare una lista per un appuntamento", "raccogli invito, documenti e domande da fare", "metti tutto in una cartellina la sera prima", "verifica il percorso e l'orario di partenza"),
    Task("vita_quotidiana", "creare una routine serale più tranquilla", "scegli due azioni semplici e ripetibili, come preparare l'occorrente e abbassare le luci", "riduci le attività che ti tengono mentalmente impegnato poco prima di dormire", "verifica dopo una settimana quali passaggi funzionano davvero"),
]

VARIANTS = (
    ("Mi spieghi come posso {goal}?", "Mi conviene fare tutto in una volta?"),
    ("Sono alle prime armi: da dove parto per {goal}?", "Qual è il passaggio che non dovrei saltare?"),
    ("Ho poco tempo. Come posso {goal} in modo essenziale?", "Che cosa posso rimandare senza creare problemi?"),
    ("Qual è un metodo semplice per {goal}?", "Come capisco se il risultato è sufficiente?"),
    ("Vorrei evitare errori comuni quando devo {goal}. Che cosa consigli?", "Quale controllo finale mi consigli?"),
    ("Puoi propormi un piano breve per {goal}?", "Come lo adatto se ho meno tempo del previsto?"),
    ("Devo {goal}, ma non so bene come organizzarmi. Mi aiuti?", "Qual è la priorità assoluta?"),
    ("Quali sono i primi passi pratici per {goal}?", "C'è un modo per renderlo più facile da mantenere?"),
    ("Vorrei {goal} senza complicarmi troppo. Come procedo?", "Come posso verificare di essere sulla strada giusta?"),
    ("Mi dai una procedura chiara per {goal}?", "Puoi riassumermi l'idea principale?"),
)


def make_dialogue(task: Task, variant: int, number: int) -> dict:
    opening, follow_up = VARIANTS[variant]
    lead = (
        f"Per {task.goal}, parti in modo semplice: {task.first_action}. "
        f"Poi {task.second_action}. Infine, {task.check}."
    )
    replies = (
        f"No. Inizia dalla parte più piccola e completa quella prima di aggiungere altro. {task.first_action.capitalize()}.",
        f"Non saltare questo: {task.second_action}. È il passaggio che rende il metodo sostenibile.",
        f"Puoi rimandare i dettagli non essenziali. Mantieni però questa priorità: {task.first_action}.",
        f"Un buon segnale è questo: {task.check}. Se succede, il metodo sta funzionando.",
        f"Fai proprio quest'ultimo controllo: {task.check}. Ti aiuta a correggere piccoli errori subito.",
        f"Riduci il piano a due passaggi: prima {task.first_action}; poi {task.second_action}.",
        f"La priorità è {task.first_action}. Il resto viene dopo, quando questa base è pronta.",
        f"Per mantenerlo semplice, ripeti sempre la stessa sequenza: {task.first_action}; poi {task.second_action}.",
        f"Controlla questo risultato: {task.check}. Se non è chiaro, torna al primo passaggio senza aggiungere complessità.",
        f"In breve: {task.first_action}; poi {task.second_action}; infine {task.check}.",
    )
    messages = [
        {"role": "user", "content": opening.format(goal=task.goal)},
        {"role": "assistant", "content": lead},
        {"role": "user", "content": follow_up},
        {"role": "assistant", "content": replies[variant]},
    ]
    if number % 5 == 0:
        messages.insert(0, {"role": "system", "content": "Rispondi in italiano in modo pratico, prudente e conciso."})
    return {
        "id": f"local-synthetic:chatgpt-v2:{number:04d}",
        "source": "local-synthetic:chatgpt-generated-v2",
        "license": "USER_APPROVED",
        "approval_status": "approved",
        "category": task.category,
        "messages": messages,
        "provenance": {
            "generator": "utils/sft/generate_italiano_chat_v2.py",
            "batch": "italiano-chat-v2",
            "task": asdict(task),
            "variant": variant + 1,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--append-to", type=Path, required=True)
    args = parser.parse_args()

    records = [
        make_dialogue(task, variant, number)
        for number, (task, variant) in enumerate(
            ((task, variant) for task in TASKS for variant in range(len(VARIANTS))), start=1
        )
    ]
    if len(records) != 500:
        raise RuntimeError(f"expected 500 records, produced {len(records)}")

    existing_ids = set()
    if args.append_to.exists():
        with args.append_to.open(encoding="utf-8") as source:
            for line in source:
                if line.strip():
                    existing_ids.add(json.loads(line)["id"])
    collisions = existing_ids.intersection(record["id"] for record in records)
    if collisions:
        raise RuntimeError(f"refusing to append duplicate IDs: {sorted(collisions)[:3]}")

    with args.append_to.open("a", encoding="utf-8") as destination:
        for record in records:
            destination.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"appended {len(records)} records to {args.append_to}")


if __name__ == "__main__":
    main()
