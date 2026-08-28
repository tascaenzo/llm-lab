# Dati del corpus

Questa directory separa i dati dal codice. I file grandi non entrano in Git: sono
scaricabili o rigenerabili e vengono descritti da manifest e documentazione.

```text
data/
  raw/       file originali, mai modificati (per esempio un dump Wikimedia)
  clean/     documenti estratti e puliti, con la loro provenienza
  derived/   file di testo pronti per il trainer del tokenizer
             e stream binari pronti per il language model
```

Le tre directory sono create dalle utility quando servono e sono ignorate da Git.

## Il flusso del pretraining

```text
fonte pubblica -> raw -> clean -> tokenizer .llmtok
                         |             |
                         +-------------+-> dataset .llmdat
conversazioni autorizzate -> clean/conversations.jsonl -> dataset .llmsft
```

- `raw` conserva il file esattamente come ricevuto; permette di ripetere la
  preparazione senza riscaricarlo.
- `clean` contiene `documents.jsonl`: una riga JSON per documento, con almeno
  `id`, `source`, `license`, `url` e `text`.
- `derived` contiene file `part-000.txt`, `part-001.txt`, ...: il solo testo
  passato a `llm-lab tokenizer train`; dopo la preparazione del language model
  contiene anche gli split `.llmdat` rigenerabili.

Non si mescolano testi anonimi in un unico file senza sapere da dove arrivano.
Ogni corpus ha inoltre un manifesto con fonti, data/versione del dump, licenza,
regole di pulizia, numero di documenti, dimensione e comando di training.

La procedura completa e gli script di estrazione e training sono in
[../docs/corpus.md](../docs/corpus.md).
La derivazione degli split del language model e' in
[../docs/dataset.md](../docs/dataset.md).
Lo schema conversazionale, i controlli di licenza e la derivazione `.llmsft`
sono in [../docs/italiano-chat-75m.md](../docs/italiano-chat-75m.md).
Il corpus chat corrente e' mantenuto direttamente in
`data/italiano-chat-corpus-v1.jsonl`; le sue regole di aggiornamento sono in
[../docs/italiano-chat-corpus-v1.md](../docs/italiano-chat-corpus-v1.md).
Il corpus SFT selezionato per il prossimo training e' invece
`data/derived/italiano-chat-corpus-v2/`; criteri, conteggi e rigenerazione
sono in [../docs/italiano-chat-corpus-v2.md](../docs/italiano-chat-corpus-v2.md).
Una spiegazione end-to-end di ogni file e del suo uso nel futuro training e' in
[../wiki/09-flusso-dati-e-artefatti.md](../wiki/09-flusso-dati-e-artefatti.md).
