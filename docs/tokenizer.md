# Tokenizer Byte-level BPE — specifica implementativa

## 1. Scopo e confini

Costruiremo un tokenizer per il testo del corpus italiano e per il language model del progetto.

**Stato di implementazione:** la v1 realizza vocabolario byte-level, pre-tokenizzazione, training BPE incrementale, `encode`/`decode`, salvataggio/caricamento `.llmtok` e comando CLI di training. Restano fuori token speciali, normalizzazione Unicode e training parallelo.

Il tokenizer ha due responsabilita':

1. **training**: osservare un corpus e produrre un insieme ordinato di merge BPE;
2. **runtime**: trasformare testo UTF-8 in ID numerici (`encode`) e invertire la trasformazione (`decode`).

La prima versione e' un **Byte-level BPE con pre-tokenizzazione deterministica**. Opera sui byte UTF-8 come dati opachi: questo garantisce che qualsiasi testo sia rappresentabile, compresi accenti, emoji, simboli e testo non italiano.

Il pre-tokenizzatore non analizza la grammatica italiana. Divide soltanto il flusso di byte in sequenze contigue di spazi, parole e punteggiatura: lettere ASCII, cifre, `_`, apostrofi e byte con valore almeno `0x80` appartengono a una parola; ogni altro segno di punteggiatura e' una sequenza autonoma. Le merge BPE non attraversano questi confini.

Non e' un obiettivo della prima versione replicare esattamente il tokenizer di GPT o Qwen. Quelli aggiungono regole linguistiche piu' elaborate, token speciali e ottimizzazioni che introdurremo solo se ne avremo una ragione misurabile.

## 2. Invarianti

Queste proprieta' devono restare vere per ogni tokenizer valido.

### 2.1 Copertura completa

I 256 valori di un byte (`0`–`255`) costituiscono il vocabolario di base. Non esiste quindi un token “sconosciuto”.

```text
ID 0..255 = un singolo byte con lo stesso valore
```

UTF-8 e' trattato come sequenza di byte, non come sequenza di caratteri Unicode. Per esempio `e'` accentata in UTF-8 occupa due byte di base prima delle eventuali fusioni BPE.

### 2.2 Round-trip esatto

Per ogni sequenza di byte valida o non valida come UTF-8:

```text
decode(encode(input)) == input
```

L'uguaglianza e' byte per byte. Il tokenizer non normalizza spazi, apostrofi, accenti, terminatori di riga o Unicode.

### 2.3 Determinismo

Con gli stessi file di input, lo stesso ordine di file, la stessa configurazione e la stessa versione del programma, il training deve produrre lo stesso file tokenizer.

In particolare:

- gli input vengono ordinati lessicograficamente per percorso;
- le coppie sono contate senza attraversare il confine tra due documenti;
- in caso di frequenza uguale vince la coppia con ID sinistro minore, poi ID destro minore;
- le merge sono applicate da sinistra a destra, senza sovrapposizioni, una regola alla volta nell'ordine in cui sono state addestrate.

### 2.4 ID stabili

Ogni merge aggiunge esattamente un token. L'ID del token creato dalla merge numero `i` e':

```text
256 + i
```

Non si riusano ID, non si rinumerano token e non si riservano token speciali nella v1.

## 3. Rappresentazione del modello

Un token BPE e' concettualmente una sequenza non vuota di byte. Il modello non deve salvare ogni sequenza esplicitamente: la sequenza si ricostruisce dai 256 token base e dalle merge precedenti.

```text
token 0..255       -> [byte corrispondente]
token 256 + i      -> bytes(token_left) concatenati a bytes(token_right)
```

### Leggere un ID nel file `.llmtok`

Il file salva le merge nell'ordine in cui sono state create. L'ordine assegna gli ID:

```text
prima merge  -> ID 256
seconda merge -> ID 257
terza merge   -> ID 258
```

Esempio concettuale per `"ciao"`:

```text
ID 256 = merge(99, 105)  -> "ci"
ID 257 = merge(256, 97)  -> "cia"
ID 258 = merge(257, 111) -> "ciao"
```

Nel file trovi le coppie numeriche, non le stringhe espanse. Questo evita di salvare due volte la stessa informazione: il loader puo' ricostruire `258 -> "ciao"` seguendo le merge precedenti.

Un ID e' quindi un indice del vocabolario. Non e' un codice Unicode e non contiene un significato linguistico autonomo. Quando il Transformer verra' implementato, quello stesso ID selezionera' una riga della matrice di embedding.

In memoria serviranno almeno queste strutture logiche:

```c
typedef struct {
    uint32_t left_id;
    uint32_t right_id;
} bpe_merge;

typedef struct tokenizer tokenizer;

typedef struct {
    uint32_t *ids;
    size_t length;
} token_sequence;
```

`tokenizer` resta opaca nell'header pubblico. Il chiamante non dipende dalla sua rappresentazione interna.

Il limite teorico di un ID e' `UINT32_MAX`; la v1 rifiuta una configurazione il cui vocabolario superi tale limite.

## 4. Formato su disco

La prima versione usa un formato binario piccolo, versionato e portabile. Il file
puo' chiamarsi, ad esempio, `italiano.llmtok`. Il manifesto laterale
`italiano.llmtok.json` resta testuale e conserva provenienza e metriche.

L'header occupa esattamente 64 byte:

| Offset | Dimensione | Campo |
|---:|---:|---|
| 0 | 8 | magic `LLMTOK\r\n` |
| 8 | 4 | versione del formato, `1` |
| 12 | 4 | dimensione header, `64` |
| 16 | 4 | vocabolario base, `256` |
| 20 | 4 | numero di merge |
| 24 | 8 | dimensione del payload |
| 32 | 32 | SHA-256 del payload |

I campi numerici sono unsigned little-endian. Il payload contiene le merge in
ordine; ciascuna occupa 8 byte:

```text
uint32 little-endian left_id
uint32 little-endian right_id
```

L'ordine nel payload assegna gli ID `256 + i`. Il loader verifica magic, versione,
dimensioni, checksum, assenza di dati aggiuntivi, riferimenti soltanto a token gia'
esistenti e assenza di coppie duplicate. Un file testuale storico o un binario
malformato viene rifiutato senza tentare un recupero silenzioso.

Il salvataggio scrive prima un file temporaneo nella stessa directory e lo rinomina
solo dopo una chiusura riuscita. Un errore di scrittura non tronca quindi un modello
valido gia' presente.

Non salviamo il corpus, la frequenza delle coppie o una tabella ridondante token→byte: il file rimane piccolo e sufficiente per `encode` e `decode`.

## 5. Algoritmo di training BPE

### 5.1 Input

Il trainer riceve un elenco di file di testo. Ogni file introduce un confine esplicito; inoltre la pre-tokenizzazione impedisce alle merge di attraversare separatori come gli spazi tra documenti. I file derivati possono quindi contenere piu' documenti separati da righe vuote. Gli input sono letti in streaming, quindi il corpus grezzo non viene mai caricato interamente in RAM.

La v1 non altera i byte in input. La pulizia, deduplicazione e selezione del corpus sono compiti precedenti al training e vengono registrati separatamente.

### 5.2 Stato iniziale

Per ogni pre-token del documento:

```text
contenuto byte: [b0, b1, b2, ...]
sequenza iniziale di token: [b0, b1, b2, ...]
```

Ogni elemento della sequenza e' memorizzato come `uint32_t`, anche se i primi 256 ID occupano meno spazio. Questo evita conversioni e rende uniforme il codice dopo le merge.

### 5.3 Un passo di training

Il trainer ripete questi passaggi fino a raggiungere `target_vocab_size` oppure finche' nessuna coppia adiacente e' disponibile:

1. Conta tutte le coppie adiacenti `(a, b)` in tutti i documenti.
2. Seleziona la coppia piu' frequente usando le regole di spareggio deterministiche.
3. Crea una nuova merge `(a, b)` e le assegna il successivo ID disponibile.
4. In ogni documento, sostituisce tutte le occorrenze non sovrapposte di `[a, b]` con il nuovo ID, scorrendo da sinistra a destra.
5. Registra la merge nel tokenizer.

Esempio per la merge `(10, 11) -> 256`:

```text
prima: [10, 11, 10, 11, 11]
dopo:  [256, 256, 11]
```

### 5.4 Aggiornamento incrementale delle frequenze

La prima raccolta visita tutti i pre-token distinti una volta e costruisce le
frequenze delle coppie. Dopo questo punto non ricontiamo piu' l'intero corpus a
ogni merge.

Per ogni coppia manteniamo:

- il conteggio corrente, pesato con la frequenza della parola;
- gli indici delle parole in cui la coppia e' comparsa;
- una generazione che identifica l'ultimo conteggio valido.

Una heap contiene candidati ordinati per frequenza decrescente e, in caso di
parita', per coppia numericamente minore. Quando estraiamo un candidato, ignoriamo
le sue copie obsolete nella heap confrontando conteggio e generazione con la tabella
corrente.

Quando scegliamo `(a, b) -> nuovo_id`, visitiamo soltanto le parole segnalate per
quella coppia. Per ogni parola realmente modificata rimuoviamo il contributo delle
sue vecchie coppie, applichiamo la merge e aggiungiamo il contributo delle nuove
coppie. Le altre parole non vengono toccate.

Le liste di parole possono contenere indici obsoleti o ripetuti: prima di applicare
la merge controlliamo che la coppia esista ancora e processiamo ogni parola una sola
volta. Questa scelta evita costose rimozioni dalle liste senza cambiare il risultato.

La heap viene ricostruita quando accumula troppe copie obsolete. E' una scansione
della tabella delle coppie, non delle sequenze del corpus, e mantiene l'uso di
memoria sotto controllo.

Il risultato resta identico al trainer precedente: stesse frequenze, stesso
spareggio e stesso ordine di merge. Cambia soltanto la quantita' di lavoro svolta.

### 5.5 Politica di memoria

| Risorsa | Gestione |
|---|---|
| Corpus originale | Letto file per file, mai mappato o caricato interamente in memoria. |
| Pre-token distinti | Conservati in RAM con frequenza e ID; i byte originali vengono liberati dopo la raccolta iniziale. |
| Frequenze delle coppie | Hash table persistente, aggiornata solo intorno alle parole modificate. |
| Selezione della prossima merge | Heap con candidati e versioni; ricostruita periodicamente. |
| Indice delle occorrenze | Liste di indici di parole per coppia; possono essere obsolete ma vengono verificate al momento dell'uso. |
| Tokenizer addestrato | Caricato interamente in RAM; un modello da 32k merge richiede solo pochi MB. |
| Testo in inferenza | Viene mantenuto solo il buffer della richiesta e la sequenza di ID risultante. |

### 5.6 Dimensione del vocabolario

La configurazione richiede:

```text
target_vocab_size >= 256
```

Il numero massimo di merge e':

```text
target_vocab_size - 256
```

Il target e' un massimo, non una garanzia: il training termina prima se tutte le sequenze sono ormai formate da un solo token e non rimangono coppie adiacenti.

### 5.7 Dimensione iniziale e misurazione

Il progetto non e' legato a uno specifico computer. Il trainer accetta gia' un
target di 32k token e il tokenizer risultante occupa solo pochi MB: possiamo quindi
partire direttamente da `target_vocab_size = 32000` quando avremo un corpus italiano
abbastanza rappresentativo.

La distinzione importante e' tra il modello prodotto e il tempo per produrlo.
Il trainer aggiorna le frequenze in modo incrementale; un training a 32k resta
comunque un lavoro importante su un corpus grande, perche' ogni merge modifica
sequenze reali. Questo e' un costo dell'algoritmo e del corpus, non un requisito di
RAM o di sistema operativo.

| Obiettivo | Corpus | Vocabolario massimo |
|---|---|---:|
| Capire e ispezionare le merge | Campione piccolo | 4k--8k |
| Primo tokenizer italiano | Corpus pulito rappresentativo | 32k |
| Corpus molto grande o training troppo lento | Stesso obiettivo | 32k, con benchmark e ulteriore ottimizzazione |

Per ogni training lo script di produzione registra durata, picco di memoria,
dimensione effettiva del vocabolario e valutazione su un campione fisso. La
valutazione include token prodotti, byte per token, round-trip e throughput. Queste
misure permettono di individuare i colli di bottiglia senza cambiare il formato
`.llmtok` ne' abbassare automaticamente l'obiettivo del progetto.

Il target resta un massimo. Se il corpus non offre abbastanza coppie utili, il
trainer si ferma prima.

## 6. Algoritmo di encode

`encode` riceve un buffer di byte e una lista di merge ordinata.

1. Applica la stessa pre-tokenizzazione deterministica usata nel training.
2. Converte i byte di ogni pre-token nel relativo ID base (`0..255`).
3. Indicizza le coppie del modello in una tabella hash che restituisce il rank della merge.
4. Cerca nella sequenza soltanto la merge disponibile con rank minimo e la applica da sinistra a destra.
5. Ripete finche' non rimangono coppie note, poi concatena i risultati e restituisce gli ID `uint32_t`.

Il risultato e' identico all'applicazione lineare di tutte le merge, ma il costo non
dipende piu' dall'intero vocabolario per ogni pre-token. La versione lineare e'
stata usata come riferimento durante la verifica del cambiamento.

## 7. Algoritmo di decode

`decode` riceve gli ID e ricostruisce i byte del token.

Per i token base emette direttamente un byte. Per un token creato da una merge, espande i due token che lo compongono nell'ordine sinistra-destra.

L'implementazione usa uno stack iterativo, invece della ricorsione C, per evitare problemi con catene profonde di merge. Se la profilazione mostrera' che il decode e' un collo di bottiglia, aggiungeremo una cache interna di token espansi. La cache non cambia il formato del tokenizer ne' il risultato.

Un ID maggiore o uguale alla dimensione del vocabolario e' un errore; non puo' essere convertito in testo.

## 8. API C implementata

L'API resta piccola e corrisponde all'header `include/tokenizer/tokenizer.h`.

```c
typedef enum {
    TOKENIZER_OK = 0,
    TOKENIZER_INVALID_ARGUMENT,
    TOKENIZER_ALLOCATION_FAILED,
    TOKENIZER_OVERFLOW,
    TOKENIZER_INVALID_TOKEN,
    TOKENIZER_IO_ERROR,
    TOKENIZER_INVALID_MODEL
} tokenizer_status;

typedef struct tokenizer tokenizer;

tokenizer_status tokenizer_create_byte_level(tokenizer **out_tokenizer);
void tokenizer_destroy(tokenizer *tokenizer);

tokenizer_status tokenizer_train(
    const char *const *input_paths,
    size_t input_count,
    uint32_t target_vocab_size,
    tokenizer **out_tokenizer);

tokenizer_status tokenizer_train_with_progress(
    const char *const *input_paths,
    size_t input_count,
    uint32_t target_vocab_size,
    tokenizer_train_progress_callback progress_callback,
    void *progress_context,
    tokenizer **out_tokenizer);

tokenizer_status tokenizer_load(const char *path, tokenizer **out_tokenizer);
tokenizer_status tokenizer_save(const tokenizer *tokenizer, const char *path);
uint32_t tokenizer_vocabulary_size(const tokenizer *tokenizer);
size_t tokenizer_merge_count(const tokenizer *tokenizer);
tokenizer_status tokenizer_encode(
    const tokenizer *tokenizer,
    const unsigned char *input,
    size_t input_length,
    token_sequence *out_tokens);
tokenizer_status tokenizer_decode(
    const tokenizer *tokenizer,
    const token_sequence *tokens,
    unsigned char **out_bytes,
    size_t *out_length);

void token_sequence_destroy(token_sequence *tokens);
void tokenizer_bytes_destroy(unsigned char *bytes);
```

Regole di proprieta':

- le stringhe e i buffer in input restano di proprieta' del chiamante;
- `tokenizer_train` e `tokenizer_load` allocano `tokenizer`; il chiamante chiama `tokenizer_destroy`;
- `tokenizer_encode` alloca `token_sequence.ids`; il chiamante chiama `token_sequence_destroy`;
- `tokenizer_decode` alloca `out_bytes`; il chiamante chiama `tokenizer_bytes_destroy`;
- tutte le funzioni che hanno un output puntatore lo impostano a `NULL` in caso di errore, quando possibile.

Per ora `tokenizer_status` appartiene al modulo tokenizer. Estrarremo un modulo `core` comune soltanto quando un secondo modulo avra' davvero bisogno degli stessi tipi e delle stesse regole.

tokenizer_train_with_progress e' la variante opzionale per un'interfaccia utente.
Notifica quattro fasi: lettura dei file, raccolta delle coppie, preparazione della
heap e merge BPE. La callback riceve lavoro completato e totale della fase; non
influenza l'ordine delle merge o il modello prodotto. tokenizer_train resta la
variante silenziosa per gli altri programmi.

## 9. File del modulo

La struttura del modulo e':

```text
include/tokenizer/tokenizer.h  API pubblica
src/tokenizer/model.c          modello BPE in memoria e append veloce delle merge dal trainer
src/tokenizer/pretokenizer.c   segmentazione deterministica dei byte
src/tokenizer/tokenizer.c      encode, decode e gestione buffer
src/tokenizer/train.c          orchestrazione del training e avanzamento
src/tokenizer/train_data.c     parole, frequenze, heap e aggiornamenti BPE incrementali
src/tokenizer/model_io.c       salvataggio e caricamento .llmtok
```

Ogni file contiene una responsabilita' effettiva; non esistono directory vuote o layer intermedi.

## 10. Limiti intenzionali della v1

- nessun token speciale (`BOS`, `EOS`, `PAD`);
- nessuna normalizzazione Unicode;
- nessuna pre-tokenizzazione con espressioni regolari o dipendenze esterne;
- nessun training parallelo;
- nessuna cache di decode;
- nessun supporto per formati diversi dalla prima versione del file modello.

Questi limiti rendono il comportamento osservabile. Le estensioni saranno aggiunte una alla volta, con una motivazione e una misura di regressione.

## 11. Criteri di completamento

Il tokenizer v1 e' pronto quando soddisfa tutti questi punti:

1. `encode` e `decode` eseguono un round-trip byte esatto su testo italiano, UTF-8 multibyte e byte arbitrari.
2. Il training sullo stesso insieme ordinato di input produce lo stesso file byte per byte.
3. Un tokenizer salvato e poi ricaricato produce gli stessi ID e lo stesso decode.
4. Un file modello malformato viene rifiutato senza crash o perdite di memoria.
5. Il programma puo' addestrare il vocabolario configurato e salvarlo in un file `.llmtok` valido.

## 12. Uso dalla CLI

```sh
llm-lab tokenizer train OUTPUT.llmtok VOCAB_SIZE INPUT...
```

Esempio:

```sh
llm-lab tokenizer train italiano.llmtok 32000 corpus-a.txt corpus-b.txt
```

Gli input vengono ordinati lessicograficamente dal trainer prima della lettura. Il comando stampa il numero effettivo di token e merge generati.

In un terminale interattivo mostra una barra per ogni fase. Durante la lettura
misura i byte effettivi dei file; durante le merge misura le regole BPE create su
VOCAB_SIZE - 256. Velocita' ed ETA sono calcolate separatamente per la fase
corrente, quindi l'ETA delle merge non viene confusa con quella della lettura. Se
l'output viene reindirizzato in un file, usa messaggi di fase su righe separate
invece di caratteri di controllo.

I tokenizer che vogliamo conservare vengono salvati in `artifacts/tokenizers/`, non in `build/`. Ogni artefatto ha un manifesto `.llmtok.json` con corpus, snapshot della provenienza, target, dimensione effettiva, merge, checksum, commit, misure del training e valutazione.

Per valutare un modello su un massimo di byte distribuiti sugli input indicati:

```sh
llm-lab tokenizer evaluate italiano.llmtok 1048576 corpus-a.txt corpus-b.txt
```

Il comando restituisce JSON, verifica anche il round-trip e puo' essere richiamato
dalla pipeline di produzione degli artefatti.

Per un esperimento rapido sul modello gia' addestrato:

```sh
tokenizer_experiment artifacts/tokenizers/italiano-wikipedia-v1.llmtok
```

Il tester riceve il percorso `.llmtok` come unico argomento, carica il modello una
sola volta e mantiene una sessione interattiva per testo → ID e ID → testo. Non
addestra ne' salva modelli: queste operazioni restano nella CLI principale.
