# Dataset del language model — specifica implementativa

## 1. Obiettivo e confini

La pipeline converte `documents.jsonl` e un tokenizer `.llmtok` in tre stream di
token pronti per il language modeling autoregressivo:

```text
documents.jsonl + tokenizer.llmtok
    -> dataset.train.llmdat
    -> dataset.validation.llmdat
    -> dataset.test.llmdat
```

La v1 esegue split deterministico per documento, decodifica delle stringhe JSON,
tokenizzazione, inserimento di `<EOD>`, scrittura binaria e lettura di batch. Non
esegue shuffle fisico dei documenti, packing avanzato, sharding multiplo o
parallelizzazione.

## 2. Input JSONL

Ogni riga non vuota deve essere un oggetto JSON e deve contenere almeno due campi
stringa:

```json
{"id":"wikipedia-it:123","text":"Testo pulito..."}
```

Gli altri campi vengono ignorati. `id` deve essere non vuoto e identifica lo split;
`text` viene decodificato secondo JSON e passato al tokenizer come UTF-8. Una riga
malformata, un ID duplicato o un campo obbligatorio assente interrompono la
preparazione senza pubblicare artefatti parziali.

## 3. Regola di split

Calcoliamo FNV-1a a 64 bit sui byte UTF-8 di `id` e usiamo `hash % 10000`:

| Intervallo | Split |
|---:|---|
| `0..8999` | training |
| `9000..9499` | validation |
| `9500..9999` | test |

La regola e' indipendente dall'ordine del JSONL e produce la proporzione attesa su
un corpus grande. Non promette che ogni split sia non vuoto per fixture molto
piccole; il comando rifiuta comunque un artefatto privo dei token necessari a
costruire almeno una coppia input/target.

## 4. Vocabolario del modello

Dato `V = tokenizer_vocabulary_size(tokenizer)`:

```text
ID <EOD>       = V
vocabolario LM = V + 1
```

La pipeline rifiuta `V == UINT32_MAX`. Dopo ogni documento codifica `text` e
aggiunge esattamente un `<EOD>`, anche per un testo vuoto. Il token speciale vive
nel dataset e nel modello, non modifica il formato `.llmtok` e non e' decodificato
come testo.

## 5. Formato `.llmdat`

Ogni file usa interi little-endian ed e' composto da un header di 128 byte seguito
da token `uint32_t` contigui.

| Offset | Dimensione | Campo |
|---:|---:|---|
| 0 | 8 | magic `LLMDATA\n` |
| 8 | 4 | versione formato, `1` |
| 12 | 4 | dimensione header, `128` |
| 16 | 4 | dimensione vocabolario tokenizer |
| 20 | 4 | dimensione vocabolario LM |
| 24 | 4 | ID `<EOD>` |
| 28 | 4 | split: `0=train`, `1=validation`, `2=test` |
| 32 | 8 | numero di token |
| 40 | 8 | numero di documenti |
| 48 | 8 | byte del payload, sempre `token_count * 4` |
| 56 | 32 | SHA-256 del file `.llmtok` completo |
| 88 | 32 | SHA-256 del payload dei token |
| 120 | 8 | riservato, zero |

Il writer crea prima `DESTINAZIONE.part`, completa payload, checksum e header,
chiude il file e infine lo rinomina. Non sovrascrive un artefatto esistente. Il
loader verifica header, dimensioni, checksum, assenza di byte extra e che ogni ID
sia minore della dimensione del vocabolario LM.

Se un'esecuzione viene interrotta, i file `.part` restano intenzionalmente sul
disco e un nuovo tentativo li segnala come output gia' esistente. La v1 non riprende
una tokenizzazione parziale: dopo avere verificato che nessun processo sia ancora
attivo, quei soli file incompleti devono essere rimossi prima di ricominciare.

## 6. Batcher

Il batcher opera direttamente sul file e non carica l'intero corpus in RAM. Per
`batch_size = B` e `context_length = T`, il chiamante fornisce due buffer da
`B * T` token.

Per ogni riga del batch viene scelto un offset nell'intervallo:

```text
0 <= offset < token_count - T
```

e vengono letti `T + 1` token:

```text
inputs [row, :] = tokens[offset     .. offset + T - 1]
targets[row, :] = tokens[offset + 1 .. offset + T]
```

Un PRNG interno con seed esplicito rende la sequenza degli offset riproducibile.
Il batcher rifiuta dataset con meno di `T + 1` token e moltiplicazioni che
eccedono `size_t`.

## 7. CLI

```sh
./build/debug/llm-lab dataset prepare \
  artifacts/tokenizers/italiano-wikipedia-v1.llmtok \
  data/clean/italiano-wikipedia-v1/documents.jsonl \
  data/derived/italiano-wikipedia-v1/lm/italiano-wikipedia-v1
```

Il prefisso finale produce:

```text
italiano-wikipedia-v1.train.llmdat
italiano-wikipedia-v1.validation.llmdat
italiano-wikipedia-v1.test.llmdat
```

La directory padre del prefisso deve esistere; la v1 crea i tre file, non la
gerarchia di directory.

Durante la lettura la CLI mostra percentuale, byte elaborati, documenti, token,
velocita' ed ETA. In un terminale interattivo aggiorna una singola barra; se
standard error viene reindirizzato, scrive un aggiornamento testuale circa ogni 5%
per lasciare un log leggibile. La percentuale misura i byte consumati dal JSONL;
la pubblicazione e la verifica finale dei tre file possono terminare poco dopo il
100%.

## 8. Verifiche richieste

- uno stesso documento finisce sempre nello stesso split;
- nessun documento compare in piu' split;
- ogni documento termina con un solo `<EOD>`;
- gli input e target del batch sono traslati di una posizione;
- stesso seed significa stessi batch;
- corruzione di header o payload viene rilevata;
- un errore non lascia file finali incompleti.
