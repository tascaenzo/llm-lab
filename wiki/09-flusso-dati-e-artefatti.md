# Flusso dei dati e artefatti prodotti

Questa pagina ricostruisce in ordine tutto cio' che il progetto ha realizzato
finora: dalla sorgente Wikipedia ai file binari che il futuro language model
leggera' durante il training.

Lo scopo non e' soltanto elencare comandi. Per ogni passaggio vogliamo sapere:

- quale problema risolve;
- quale file riceve in input;
- quale file produce;
- quale componente usera' quel risultato;
- se il file e' sorgente, dato derivato o artefatto versionato.

Le specifiche di dettaglio restano in [docs/corpus.md](../docs/corpus.md),
[docs/tokenizer.md](../docs/tokenizer.md) e
[docs/dataset.md](../docs/dataset.md).

## 1. Il flusso completo

```text
Wikimedia
   |
   v
dump XML compresso + source.json                 data/raw/wikipedia-it/
shard Parquet + source.json                      data/raw/fineweb2-ita/
   |
   | download, verifica, pulizia e normalizzazione
   v
un documents.jsonl per fonte                     data/clean/
   |
   | deduplicazione fra le fonti  <- prima della divisione, non dopo
   | composizione secondo le quote
   v
documents.jsonl + manifest.json                  data/clean/
   |
   +-------------------------------+
   |                               |
   | testo in parti                 | documenti con ID e metadati
   v                               v
part-000.txt ... part-017.txt       split 90% / 5% / 5%
   |                               |
   | training BPE                  | tokenizzazione + <EOD>
   v                               v
tokenizer.llmtok                    train/validation/test.llmdat
   |                               |
   +---------------+---------------+
                   |
                   | batch input/target
                   v
             training del modello                 non ancora implementato
                   |
                   v
             checkpoint del modello               futuro artifacts/models/
```

Il tokenizer e il dataset sono due prodotti diversi dello stesso corpus:

- il tokenizer impara **come rappresentare** il testo con ID compatti;
- il language model imparera' **quali token seguono altri token**.

## 2. Le categorie di file

La posizione di un file descrive il suo ruolo.

| Directory | Significato | Entra in Git? |
|---|---|---|
| `data/raw/` | Sorgenti originali immutabili | No: troppo grandi e riscaricabili |
| `data/clean/` | Documenti puliti con provenienza | No: grandi e rigenerabili |
| `data/derived/` | Viste binarie o testuali ottenute dai dati puliti | No: grandi e rigenerabili |
| `artifacts/` | Modelli piccoli e manifesti che identificano un esperimento | Si', quando ragionevole |
| `build/` | Eseguibili e file temporanei di compilazione | No: ricostruibili dal codice |

Un `.llmdat` e' un artefatto nel senso generale di “risultato della pipeline”, ma
rimane in `data/derived/` perche' occupa diversi GiB ed e' ricostruibile. Il
tokenizer `.llmtok`, invece, occupa circa 248 KiB e viene conservato in
`artifacts/tokenizers/` insieme al suo manifesto.

## 3. Catalogo dei file attuali

| File | Prodotto da | Contenuto | Consumato da | Ruolo durante il training |
|---|---|---|---|---|
| `data/raw/wikipedia-it/itwiki-20260801-pages-articles.xml.bz2` | downloader | Dump Wikipedia originale compresso | estrattore | Non viene letto dal modello; permette di rigenerare il corpus |
| `data/raw/wikipedia-it/source.json` | downloader | URL, data, licenza e checksum della fonte | estrattore e audit umano | Dimostra da dove arrivano i dati |
| `data/raw/fineweb2-ita/*.parquet` | `download_fineweb.py` | Shard web di FineWeb-2 italiano | normalizzatore | Una shard rende gia' piu' token di quanti ne servano |
| `data/clean/<corpus>/deduplication.json` | `deduplicate_corpus.py` | Quanti documenti sono stati scartati e per colpa di quale fonte | audit umano | E' la prova che validation e test non contengono copie di train |
| `data/clean/italiano-wikipedia-v1/documents.jsonl` | estrattore | Un documento pulito per riga, con ID e metadati | preparatore del dataset | Sorgente canonica da cui derivano gli split |
| `data/clean/italiano-wikipedia-v1/manifest.json` | estrattore | Configurazione, filtri, statistiche e percorsi | trainer del tokenizer e audit | Rende riproducibile la selezione del corpus |
| `data/derived/italiano-wikipedia-v1/tokenizer-train-input/part-*.txt` | derivatore train-only | Solo testo dello split train | trainer BPE | Validation e test non influenzano il vocabolario |
| `artifacts/tokenizers/italiano-wikipedia-v2.llmtok` | trainer BPE | 256 byte token e 31.744 merge | preparatore dataset, generazione futura | Converte testo in ID e ID testuali in byte |
| `artifacts/tokenizers/<corpus>.llmtok` | trainer BPE | Vocabolario del corpus multi-sorgente | preparatore dataset | Il vocabolario del modello aggiunge `<EOD>` e sette slot riservati |
| `artifacts/tokenizers/italiano-wikipedia-v2.llmtok.json` | wrapper del trainer | Checksum, split train, comando, commit e metriche | audit e riproduzione | Impedisce di confondere tokenizer con provenienze diverse |
| `*.train.llmdat` | `dataset prepare` | Stream dei token di training | batcher del trainer futuro | Produce esempi che aggiornano i pesi |
| `*.validation.llmdat` | `dataset prepare` | Stream dei token di validation | ciclo di valutazione futuro | Misura la loss senza modificare i pesi |
| `*.test.llmdat` | `dataset prepare` | Stream dei token di test | valutazione finale futura | Misura il modello soltanto dopo le decisioni di sviluppo |
| `*.llmdat.part` | `dataset prepare` interrotto | File incompleto, senza pubblicazione finale | nessuno | Deve essere rimosso prima di ricominciare; non e' un dataset valido |
| checkpoint futuro | trainer del modello | Pesi, optimizer, step, seed e configurazione | ripresa e inferenza | Conserva cio' che il modello ha imparato |

### Perche' esistono sia `documents.jsonl` sia `part-*.txt`?

`documents.jsonl` conserva i confini e la provenienza dei documenti. Questo e'
necessario per assegnare una pagina intera a train, validation oppure test.

I `part-*.txt` sono invece una vista semplificata per il trainer BPE: contengono
solo testo e possono essere letti in streaming. Non devono essere usati per creare
gli split del language model, perche' hanno perso gli ID e gli altri metadati.

## 4. Dalla fonte al corpus pulito

Il downloader ha salvato uno snapshot preciso di Wikipedia in italiano:

```text
data/raw/wikipedia-it/
  itwiki-20260801-pages-articles.xml.bz2    circa 3,9 GiB
  source.json                              529 byte
```

`source.json` registra sia il checksum pubblicato dalla fonte sia lo SHA-256
calcolato localmente. L'alias remoto `latest` non entra nella provenienza finale:
il file locale punta allo snapshot datato `20260801`.

L'estrattore ha poi:

1. visitato 4.127.796 pagine grezze;
2. considerato soltanto il namespace principale;
3. scartato 1.128.966 redirezioni;
4. scartato 307.590 testi troppo corti;
5. rimosso 912 duplicati esatti;
6. scritto 1.672.132 documenti puliti.

Il risultato canonico e':

```text
data/clean/italiano-wikipedia-v1/
  documents.jsonl    circa 4,6 GiB
  manifest.json
```

Ogni riga di `documents.jsonl` e' indipendente:

```json
{"id":"wikipedia-it:123","source":"wikipedia-it","license":"CC BY-SA","url":"https://it.wikipedia.org/?curid=123","title":"Titolo","text":"Testo pulito..."}
```

`id` e' particolarmente importante: determina lo split senza dipendere
dall'ordine delle righe.

## 5. Dal corpus al tokenizer

Durante la stessa estrazione e' stata creata una vista contenente soltanto testo:

```text
data/derived/italiano-wikipedia-v1/tokenizer-train-input/
  part-000.txt
  ...
  part-017.txt
```

Le 18 parti occupano complessivamente circa 4,3 GiB. La suddivisione limita la
dimensione dei singoli file; non cambia il risultato del training BPE.

Il trainer ha costruito:

```text
artifacts/tokenizers/
  italiano-wikipedia-v2.llmtok
  italiano-wikipedia-v2.llmtok.json
```

Il tokenizer risultante contiene:

- 256 token base, uno per ogni possibile byte;
- 31.744 merge BPE;
- 32.000 token testuali totali;
- round-trip byte per byte verificato;
- circa 2,76 byte di testo per token sul campione di valutazione.

Il `.llmtok` e' il dato operativo. Il `.llmtok.json` spiega come e' stato creato e
include lo snapshot del manifesto del corpus. Se il file binario cambia, cambia
anche il significato degli ID e quindi deve cambiare l'intero esperimento.

## 6. Dal corpus agli split del language model

Il comando `dataset prepare` legge insieme:

```text
documents.jsonl + tokenizer.llmtok
```

Per ogni documento:

1. calcola lo split dall'hash del suo `id`;
2. converte `text` in token BPE;
3. aggiunge `<EOD>` per segnare la fine del documento;
4. scrive i token nello stream dello split scelto.

Con il tokenizer attuale:

```text
token testuali: 0..31999
<EOD>:          32000
vocabolario LM: 32001
```

Gli artefatti reali prodotti sono:

| Split | Documenti | Token, incluso `<EOD>` | Dimensione approssimativa |
|---|---:|---:|---:|
| training | 1.503.988 | 1.560.387.976 | 5,8 GiB |
| validation | 84.030 | 86.224.269 | 329 MiB |
| test | 84.114 | 86.330.188 | 329 MiB |
| **Totale** | **1.672.132** | **1.732.942.433** | **6,5 GiB** |

Il totale dei documenti coincide con `documents.jsonl`: nessuna pagina e' stata
persa e nessuna pagina appartiene a piu' split.

### Perche' i `.llmdat` sono piu' grandi del testo?

Ogni ID viene salvato come `uint32_t`, quindi occupa quattro byte. BPE riduce il
numero di elementi rispetto ai byte originali, ma la rappresentazione numerica e'
pensata per essere letta rapidamente dal trainer, non per comprimere il corpus.

Ogni `.llmdat` contiene inoltre un header con:

- versione del formato;
- split;
- numero di documenti e token;
- dimensione del vocabolario;
- ID `<EOD>`;
- SHA-256 del tokenizer;
- SHA-256 del payload.

Il checksum registra con quale tokenizer e' stato creato lo stream e permettera'
al trainer di rifiutare combinazioni incompatibili prima di usare gli ID.

## 7. Come i tre split verranno usati

I tre file non sono intercambiabili.

### Training

Il trainer legge finestre casuali da `train.llmdat`. Per ogni batch esegue:

```text
token -> forward -> loss -> backward -> aggiornamento dei pesi
```

Questo e' l'unico split autorizzato a modificare i parametri del modello.

### Validation

Periodicamente il trainer sospende gli aggiornamenti e legge alcuni batch da
`validation.llmdat`:

```text
token -> forward -> loss -> registrazione della metrica
```

Non esegue backward e non modifica i pesi. La validation serve per confrontare
configurazioni, scegliere quando fermarsi e riconoscere l'overfitting.

Poiche' influenza le decisioni di sviluppo, la validation non e' una misura finale
completamente indipendente.

### Test

`test.llmdat` deve rimanere chiuso durante il normale sviluppo. Viene usato dopo
avere scelto architettura, iperparametri e checkpoint. Non esegue backward e non
deve guidare ulteriori modifiche; altrimenti diventerebbe di fatto un secondo
validation set.

## 8. Da uno stream a input e target

Il batcher non salva coppie duplicate sul disco. Le ricostruisce spostando di una
posizione una finestra di `T + 1` token:

```text
stream:  [11, 22, 33, 44, 55]
input:   [11, 22, 33, 44]
target:  [22, 33, 44, 55]
```

Il modello ricevera' `input` e dovra' assegnare alta probabilita' a `target` in
ogni posizione. La risposta corretta non viene annotata manualmente: e' il token
successivo gia' presente nel testo.

Una finestra puo' contenere:

```text
... ultimo token del documento, <EOD>, primo token del documento seguente ...
```

Questo insegna al modello che un documento puo' terminare. Durante la generazione,
`<EOD>` non viene passato al decoder testuale: puo' arrestare il campionamento o
separare due documenti.

## 9. Il ciclo futuro del trainer

Il trainer non e' ancora implementato. Quando lo sara', il flusso minimo previsto
e':

```text
apri train.llmdat
apri validation.llmdat
inizializza modello e optimizer

ripeti per ogni step:
    batch = prossimo batch di training
    logits = forward(batch.inputs)
    loss = cross_entropy(logits, batch.targets)
    gradienti = backward(loss)
    aggiorna i pesi

    periodicamente:
        calcola validation loss senza backward
        salva un checkpoint

alla fine:
    scegli il checkpoint senza guardare il test
    calcola una volta la test loss
```

Un checkpoint futuro dovra' contenere almeno:

- pesi del modello;
- stato dell'optimizer;
- step corrente;
- seed e stato casuale;
- configurazione dell'architettura;
- checksum o identita' del tokenizer e del dataset.

## 10. Riproducibilita' e controlli

Ogni livello risponde a una domanda diversa:

| Controllo | Domanda a cui risponde |
|---|---|
| checksum del dump | Abbiamo scaricato i byte attesi? |
| manifesto del corpus | Quali filtri e regole hanno prodotto i documenti? |
| ID del documento | Lo split rimane stabile se cambia l'ordine del file? |
| checksum `.llmtok` | Gli ID significano gli stessi token? |
| checksum `.llmdat` | Lo stream di token e' integro? |
| seed del batcher | Possiamo ricostruire la sequenza dei batch? |
| checkpoint | Possiamo riprendere lo stesso training? |

I dati grandi non entrano in Git, ma il codice, le specifiche e i manifesti piccoli
permettono di capire e ripetere le trasformazioni.

## 11. Interruzioni e file `.part`

Durante la preparazione del dataset si scrive prima:

```text
DESTINAZIONE.llmdat.part
```

Solo dopo avere completato payload, header e checksum il file viene rinominato in
`.llmdat`. Se il processo viene interrotto, `.part` resta sul disco e non deve
essere usato per il training.

La v1 non supporta la ripresa. Prima di rimuovere un `.part` bisogna verificare che
la preparazione non sia ancora attiva; poi si elimina soltanto il file incompleto e
si rilancia lo stesso comando.

## 12. Training generale e training tematico

Il dataset attuale rappresenta italiano enciclopedico generale. Per specializzare
in seguito il modello su un argomento:

1. si crea un nuovo `documents.jsonl` con fonti e licenze registrate;
2. si riusa normalmente lo stesso `.llmtok`, per mantenere compatibili gli ID e i
   checkpoint;
3. si genera un nuovo trio `.llmdat` con un prefisso diverso;
4. si continua il training da un checkpoint generale;
5. si mescolano anche batch generali per limitare il catastrophic forgetting;
6. si misurano separatamente validation generale e validation tematica.

Un corpus tematico non sostituisce automaticamente il corpus generale. Risponde a
un obiettivo diverso: adattare un modello che ha gia' imparato strutture linguistiche
piu' ampie.

## 13. Stato reale del progetto

### Completato

- download e verifica dello snapshot Wikipedia;
- estrazione, pulizia e deduplicazione;
- manifesto del corpus;
- trainer Byte-level BPE;
- formato `.llmtok`, caricamento, encoding e decoding;
- split deterministico per documento;
- formato `.llmdat` con checksum;
- token `<EOD>`;
- batcher casuale riproducibile;
- runtime CPU di riferimento, tensori e operazioni numeriche;
- CLI con avanzamento;
- test unitari, integrazione e sanitizer.

### Modello implementato e prossimo gate

- embedding, forward, backward, registry dei parametri e AdamW nel trainer;
- checkpoint atomici, ripresa, valutazione su validation e generazione greedy;
- il Modello Minimal: un blocco Transformer causale CPU e un baseline diagnostico.

La prossima milestone non consiste nel raccogliere altri dati o far crescere
subito il numero di parametri. Consiste nella parita' del training del Modello
Minimal su Metal:
RMSNorm, RoPE, attention causale, backward e AdamW devono produrre risultati
coerenti con CPU. Il Modello Minimal e' il riferimento piccolo; la
configurazione scalabile multi-layer/multi-head/SwiGLU seguira' dopo la parita'
Metal. Il piano e i gate sono descritti
in [primo modello addestrabile](13-primo-modello-addestrabile.md).

## 14. Domande di controllo

Se questa pagina e' chiara, dovresti saper rispondere a queste domande:

1. Perche' `documents.jsonl` conserva gli ID mentre `part-*.txt` no?
2. Perche' il tokenizer e il language model vengono addestrati separatamente?
3. Perche' `<EOD>` ha ID 32000 ma non appartiene al `.llmtok`?
4. Perche' soltanto `train.llmdat` puo' aggiornare i pesi?
5. Perche' non scegliamo gli iperparametri guardando `test.llmdat`?
6. Perche' i `.llmdat` restano in `data/derived/` e non in Git?
7. Quali checksum impediscono di combinare file incompatibili o corrotti?
8. Quale componente trasforma una finestra di token in input e target?
