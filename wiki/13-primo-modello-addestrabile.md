# Dal primo modello addestrabile al decoder scalabile

## Stato attuale

M0 e' completato come baseline CPU: embedding, output head, cross-entropy,
backward e AdamW. M1 e' il primo Transformer causale CPU: aggiunge RMSNorm,
Q/K/V, RoPE, attention, proiezione e residual, con checkpoint, ripresa,
generazione greedy e valutazione sullo split validation.

M1 e' un riferimento intenzionalmente piccolo e ristretto: un solo layer, una
head e nessun MLP. Ha gia' dimostrato che il training riduce loss e validation
loss su testo italiano; non e' ancora un chatbot o un modello editoriale.
Prima di run CPU molto lunghi, il prossimo gate e' riprodurre il suo training
step su Metal entro tolleranze numeriche definite.

## Fondazione: il primo incremento completato

Il prossimo passo non e' un prototipo finto: e' il primo **language model autoregressivo reale** del progetto. Riceve token, produce logits per il token successivo, calcola cross-entropy, propaga i gradienti e aggiorna parametri appresi con AdamW.

La prima architettura, M0, era intenzionalmente piccola:

```text
token IDs [B,T]
  -> token embedding [B,T,C]
  -> proiezione lineare [B*T,C] x [C,V]
  -> logits [B*T,V]
  -> cross-entropy con target [B*T]
```

- `B` e' il batch size;
- `T` e' la lunghezza di contesto;
- `C` e' `hidden_size`;
- `V` e' il vocabolario del language model, incluso `<EOD>`.

M0 non contiene attention, RoPE o blocchi Transformer. Impara davvero una
distribuzione del token successivo, ma non combina in modo ricco l'intero
contesto. E' stato il primo gradino verificabile verso il decoder-only
Transformer M1.

## Perche' iniziare piccolo

Un modello piccolo permette di isolare gli errori prima che siano nascosti nella complessita' dell'attention e di molti layer. Deve pero' usare gli stessi concetti che resteranno nel modello finale:

- parametri F32 e gradienti F32 distinti;
- stesso dataset `.llmdat` e stesso batcher autoregressivo;
- stessa loss cross-entropy e stesso aggiornamento AdamW;
- stessa astrazione `llm_backend`;
- configurazione e formato dei parametri estendibili.

Non si riscrivera' quindi il modello quando verranno aggiunti Transformer, RMSNorm e attention: si aggiungeranno parametri, attivazioni e passaggi al modello esistente.

## CPU prima, stessa architettura per Metal

Il codice del modello non deve conoscere CPU o Metal:

```text
model / trainer
       |
       v
runtime API llm_*
       |
       +-- CPU: riferimento numerico completo
       `-- Metal: accelerazione delle primitive disponibili
```

Il primo training viene eseguito su CPU per una ragione pratica: la CPU ha gia' tutte le primitive runtime v1, inclusi AdamW, RMSNorm, RoPE e attention con backward. Metal e' gia' valido per memoria, batch, GEMM, embedding, cross-entropy, `accumulate` e SiLU, ma non ha ancora RMSNorm, RoPE, attention e AdamW.

Questo non crea due modelli. Il modello CPU e il futuro modello Metal usano le stesse strutture, la stessa sequenza di operazioni e gli stessi checkpoint. La CPU serve come riferimento per confrontare in seguito logits, loss, gradienti e parametri dopo un update Metal entro tolleranze dichiarate.

## Configurazione ed evoluzione scalabile

La configurazione descrive il modello estendibile, anche se M1 non rende ancora
tutti i campi configurabili:

| Campo | Ruolo iniziale | Esempio smoke test |
|---|---|---:|
| `vocabulary_size` | derivato dal dataset, incluso `<EOD>` | dal `.llmdat` |
| `context_length` | token per esempio | 32 |
| `hidden_size` | dimensione embedding e hidden state | 64 |
| `layer_count` | zero in M0, uno in M1; numero di blocchi in M2 | 0 / 1 / 4 |
| `head_count` | zero in M0, una in M1; teste per blocco in M2 | 0 / 1 / 4 |
| `feed_forward_size` | zero in M1; dimensione SwiGLU in M2 | 0 / 1024 |
| `seed` | inizializzazione riproducibile | valore esplicito |

M0 richiede `layer_count=head_count=feed_forward_size=0`; M1 richiede
`layer_count=1`, `head_count=1`, `feed_forward_size=0`. M2 rendera' questi
assi realmente configurabili: una pila di blocchi decoder-only, multi-head e
SwiGLU. Il registry dinamico dei parametri, il trainer e il checkpoint restano
gli stessi; non verranno creati due modelli separati.

Il costo cresce in modo diverso a seconda della configurazione: i layer sono
approssimativamente lineari, molte matrici crescono quadraticamente con
`hidden_size`, mentre l'attention cresce circa con il quadrato del contesto.
Per questo un primo target M2 da circa 20M parametri va verificato prima con
Metal e poi con training prolungato, non dedotto soltanto aumentando un flag.

## Parametri e gradienti

Per il modello minimo servono due parametri appresi:

| Nome | Shape | Forward | Backward |
|---|---|---|---|
| `token_embedding` | `[V,C]` | `gather_rows(token IDs)` | `scatter_add_rows` del gradiente hidden |
| `output_weight` | `[C,V]` | hidden `@ output_weight` | `d_hidden = d_logits @ output_weight^T`; `d_weight = hidden^T @ d_logits` |

Ogni parametro deve possedere anche gradiente, primo e secondo momento AdamW, nome stabile e shape verificabile. Il peso di output resta separato dall'embedding nella prima versione; weight tying e' un'ottimizzazione futura che non deve complicare il primo backward.

## Sequenza di un training step

```text
1. lm_batcher_next() -> input IDs e target IDs
2. azzera i gradienti del modello
3. embedding lookup                         -> hidden [B,T,C]
4. reshape hidden                            -> [B*T,C]
5. matmul output                             -> logits [B*T,V]
6. cross entropy                             -> loss scalare
7. backward cross entropy                    -> d_logits [B*T,V]
8. backward linear                           -> d_hidden, d_output_weight
9. scatter-add embedding                     -> d_token_embedding
10. AdamW su ogni parametro e sui suoi momenti
11. registra loss e step
```

Il trainer possiede un registry di parametri, lo step e gli iperparametri comuni. Il runtime aggiorna un parametro per chiamata; non deve esistere logica AdamW duplicata per ciascun peso.

## Ordine di implementazione

1. Creare il modulo `model` con config, strutture parametro, inizializzatore deterministico e distruzione ordinata dei tensori.
2. Implementare embedding forward/backward e testarne il gradiente.
3. Implementare linear forward/backward usando `llm_matmul_ex`.
4. Creare registry parametri e funzioni `zero_grad` e `optimizer_step`.
5. Collegare il batcher `.llmdat` e implementare un training step CPU.
6. Aggiungere CLI `model train` con seed, batch size, contesto, hidden size, learning rate e numero di step espliciti.

Checkpoint, ripresa, generazione greedy e valutazione sono stati aggiunti a
M0/M1. Scheduler, clipping e checkpoint periodici restano nella milestone di
training affidabile.

## Gate di correttezza

Il modello minimo e' completato quando tutti i seguenti punti sono veri:

- gradient check a differenze finite per embedding e proiezione lineare;
- test di forme e ownership per `B=1`, `T=1`, vocabolario piccolo e batch irregolari;
- stessa seed produce gli stessi parametri e la stessa sequenza di loss CPU;
- un corpus/fixture minuscolo viene overfittato: la training loss cala in modo netto e ripetibile;
- il modello resta backend-agnostic: non contiene `#if` CPU/Metal ne' oggetti Metal.

L'overfit e' un test fondamentale, non l'obiettivo finale di qualita': dimostra che dati, forward, loss, backward, registry e optimizer sono collegati correttamente.

## Evoluzione senza riscrittura

```text
M0  embedding -> head -> logits                          baseline completata
M1  + RMSNorm -> attention causale -> residual           riferimento CPU completato
M1M Metal training M1: forward/backward/AdamW            prossimo gate
M2  + multi-head + SwiGLU + piu' blocchi                 decoder configurabile
M3  scheduler, clipping, checkpoint periodici            training affidabile
```

Metal viene completato ora perche' M1 fornisce forme reali (`B*T`, `C`, `V`) e
un riferimento CPU con cui scegliere e verificare i kernel da ottimizzare.

Prosegui con [architettura Transformer](03-architettura-transformer.md) per i blocchi che verranno aggiunti dopo M0, e con [addestramento](04-addestramento.md) per il significato della loss e del ciclo di training.
