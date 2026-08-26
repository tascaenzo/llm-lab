# Italiano-Base-75M — specifica del primo decoder utilizzabile

**Stato:** il decoder configurabile e il trainer per run lunghi sono
implementati. La parita' CPU/Metal copre una configurazione multi-layer,
multi-head e SwiGLU; un update della configurazione canonica 75M e' stato
eseguito con successo su Metal. Il trainer accumula micro-batch, usa warmup
lineare seguito da cosine decay, applica clipping a norma globale, valuta uno
split separato e salva checkpoint latest/best con log JSONL. Resta da eseguire
il training lungo del checkpoint e valutarne la qualita' linguistica.

`Italiano-Base-75M` e' il primo modello del progetto progettato per produrre
completamenti italiani brevi utili, non solo per dimostrare il runtime. Estende
il [Modello Minimal](model-minimal.md) senza creare una seconda famiglia di
codice: conserva API `lm_model`, trainer, checkpoint e backend CPU/Metal.

Il target e' il Mac mini Apple M4 con 24 GiB di memoria unificata usato per il
Modello Minimal. Il vincolo non e' riempire tutta la RAM con il massimo numero
di parametri: i 2.557.390.263 token dello split train di `italiano-v3`
determinano una dimensione di modello che puo' essere addestrata con dati
sufficienti e verificata localmente.

## Obiettivo di prodotto

Il checkpoint base deve:

- completare frasi e paragrafi brevi in italiano corretto;
- proseguire testo informativo e narrativo per una finestra di 512 token;
- rispettare prompt brevi e non ripetere meccanicamente gli stessi token;
- essere valutabile e riproducibile con gli artefatti del progetto.

Non e' ancora un assistente conversazionale affidabile. Wikipedia insegna
italiano generale e registro enciclopedico, non il comportamento domanda /
risposta. Un successivo fine-tuning supervisionato su dialoghi italiani e'
necessario per il prodotto `Italiano-Chat-75M`.

## Configurazione canonica

| Campo | Valore | Motivazione |
|---|---:|---|
| vocabolario LM | 32.008 | tokenizer train-only `italiano-v3`, `<EOD>` e 7 slot SFT riservati |
| contesto `T` | 512 | permette completamenti piu' lunghi del riferimento a 32 token |
| hidden size `C` | 512 | ampiezza adatta al budget del corpus e dell'hardware |
| layer `L` | 12 | profondita' sufficiente per dipendenze linguistiche non locali |
| attention head `H` | 8 | `head_dimension = C / H = 64` |
| feed-forward `F` | 1.608 | MLP SwiGLU scelto per portare il totale a 75,01M parametri |
| precisione | F32 | contratto runtime v1; nessun cast o mixed precision |
| output embedding | non condiviso | prima implementazione semplice e coerente col Modello Minimal |

La configurazione e' valida soltanto se `C % H == 0`, `F > 0`, `L > 0` e il
contesto del trainer coincide con quello del modello. RoPE non aggiunge
parametri addestrabili.

### Conteggio dei parametri

La prima implementazione usa proiezioni senza bias e una RMSNorm prima di
attention, una prima di MLP e una RMSNorm finale:

```text
embedding                     V * C                  = 16.388.096
output head                   C * V                  = 16.388.096
attention per layer           4 * C * C              =  1.048.576
SwiGLU per layer              3 * C * F              =  2.469.888
RMSNorm                       (2 * L + 1) * C        =     12.800
---------------------------------------------------------------
totale                                                 75.010.560
```

Il nome `75M` identifica questa configurazione entro lo 0,014%; non deve essere
usato da solo per identificare una milestone Metal o un checkpoint.

## Architettura del decoder

Per ogni blocco `l` della pila, con input `x` di forma `[B,T,C]`:

```text
n = RMSNorm_attn_l(x)
q, k, v = linear_qkv_l(n)
q, k = RoPE(q, k)                         # [B,T,H,C/H]
a = causal_multi_head_attention(q, k, v)
x = x + linear_out_l(a)

n = RMSNorm_mlp_l(x)
gate, up = linear_gate_l(n), linear_up_l(n)
x = x + linear_down_l(SiLU(gate) * up)
```

Dopo l'ultimo blocco:

```text
logits = output_head(RMSNorm_final(x))    # [B*T,V]
```

Il backward attraversa i blocchi in ordine inverso e accumula un solo
gradiente per ogni parametro. I parametri devono restare in un registry
dinamico, con nomi stabili per layer e componente; il checkpoint continua a
serializzare configurazione, valori, gradienti AdamW e stato del batcher.

### Confini di implementazione

- `lm_model_config` diventa veramente configurabile per `L`, `H` e `F`; non si
  aggiunge un modello parallelo o un percorso speciale per 75M.
- Le tabelle RoPE hanno forma `[T,C/H/2]`, non `[T,C/2]`.
- Attention usa inizialmente lo stesso numero di head Q, K e V. GQA resta una
  capacita' del runtime, non una modifica necessaria a questo target.
- Il trainer e i layer non contengono oggetti Metal; l'intero modello usa solo
  l'API runtime pubblica.
- La KV cache e il decode incrementale sono fuori da questa milestone: la
  generazione puo' ricomputare la finestra di 512 token. Saranno ottimizzati
  quando esistera' un checkpoint base utile da servire.

## Budget di memoria e batch

F32 con AdamW mantiene per ogni parametro valore, gradiente, primo e secondo
momento: il solo stato parametrico richiede circa 1,12 GiB. Con micro-batch 4
e contesto 512, logits e gradiente dei logits occupano circa 250 MiB ciascuno.
Attivazioni, gradienti intermedi, attention e buffer Metal aumentano il picco;
il loro valore effettivo deve essere misurato, non stimato come disponibilita'
garantita.

Il pilot canonico del 14 agosto 2026 ha completato un update su Metal in 2,57
secondi, presentando 4.096 token a circa 1.593 token/s e registrando 3,87 GiB
di buffer attivi/picco. E' una misura di fattibilita', non una previsione
garantita del throughput di un run lungo.

Il 20 agosto 2026 il medesimo checkpoint e' stato ripreso senza conversioni su
una RTX 4090 CUDA: 1.000 update, dallo step 86.044 allo step 87.044, hanno
confermato checkpoint, validation e best-checkpoint operativi sul nuovo
backend. Rispetto al piano di 624.362 update, il run e' arrivato al 13,94%.
La velocita' osservata nel run lungo e' stata circa 5,72 step/s contro
circa 0,64 step/s sul Mac mini M4 con Metal: un vantaggio operativo di circa
8,9x. Alla fine della sessione la validation era loss `2.88134766` e perplexity
`17.83829688`; il confronto con il valore Metal precedente non e' un benchmark
paritario, perche' il modello aveva nel frattempo ricevuto ulteriori update.
I dettagli della verifica CUDA e della comparazione sono in
[Backend CUDA](backend-cuda.md).

La configurazione iniziale e':

```text
micro_batch_size = 4
context_length   = 512
gradient_accumulation_steps = 2
effective_batch_tokens = 4 * 512 * 2 = 4.096
```

Il trainer deve ridurre il micro-batch se le metriche Metal mostrano pressione
sulla memoria unificata. Non deve affidarsi a swap, memory pressure o fallbacks
CPU per completare uno step.

## Training del modello base

Il batcher predefinito percorre una permutazione riproducibile dei blocchi
contigui non sovrapposti di 512 token: ogni blocco compare una sola volta
nell'epoca, senza caricare l'intero stream llmdat in RAM. Lo stato della
permutazione e' nel checkpoint. La permutazione e' affine — offset iniziale
casuale piu' passo coprimo con il numero di blocchi — quindi copre tutta l'epoca
esattamente una volta con memoria costante, ma non e' uniforme sullo spazio
delle permutazioni: i blocchi di uno stesso batch restano equidistanti nel
corpus. E' una proprieta' accettata in cambio dello streaming; una permutazione
uniforme richiederebbe di materializzare tre milioni di indici.
L'opzione --sampling random resta disponibile solo per compatibilita' o
esperimenti brevi.

Il trainer disponibile espone il micro-batch come --batch-size e il numero di
micro-batch per update come --gradient-accumulation. Per esempio, la
configurazione iniziale e':

    ./build/release/llm-lab model train DATASET.train.llmdat 10000 \
      --backend metal --context 512 --hidden 512 --layers 12 --heads 8 --ffn 1608 \
      --batch-size 4 --gradient-accumulation 2 \
      --learning-rate 3e-4 --min-learning-rate 3e-5 \
      --warmup-steps 8000 --total-steps 624362 \
      --beta1 0.9 --beta2 0.95 --epsilon 1e-8 --weight-decay 0.1 \
      --sampling shuffled --gradient-clip 1.0 \
      --checkpoint artifacts/models/italiano-base-75m/latest.llmckpt \
      --checkpoint-every 2000 \
      --validation DATASET.validation.llmdat --validation-every 2000 \
      --validation-batches 100 \
      --best-checkpoint artifacts/models/italiano-base-75m/best.llmckpt \
      --log artifacts/models/italiano-base-75m/training.jsonl

STEPS, warmup e decay contano update di AdamW, non micro-batch. Il checkpoint
viene scritto in modo atomico allo stesso percorso ogni intervallo e puo'
essere ripreso con --resume; il formato v4 conserva anche accumulo, scheduler,
sampler a blocchi e clipping. I checkpoint precedenti restano leggibili e i v3
mantengono il sampler storico per una ripresa esatta.

A circa 1,75 secondi per update su RTX 4090 (circa 5,72 step/s), 624.362 update
richiedono circa 1,3 giorni di GPU continua; sul Mac mini M4 a circa 0,64
step/s il medesimo ordine di grandezza e' circa 11,3 giorni. La cadenza dei
checkpoint e' la finestra di lavoro che un crash puo'
distruggere. Con --checkpoint-every 2000 quella finestra vale circa settanta
minuti e ogni scrittura costa circa 859 MiB (valori piu' i due momenti AdamW),
quindi il costo in I/O resta trascurabile rispetto al rischio. `SIGINT` e `SIGTERM`
sono gestiti: il comando completa lo step in corso, esegue la validation
finale, salva il checkpoint, registra un evento `interrupted` nel log JSONL e
termina con esito zero. Ctrl-C e' quindi il modo corretto di fermare un run.

Un'epoca contiene 4.994.902 blocchi completi. Con otto blocchi per update, i
624.362 update completi presentano 2.557.386.752 target token e lasciano fuori
solo 3.511 token. Il primo piano sperimentale e':

| Parametro | Valore iniziale | Regola |
|---|---:|---|
| token budget | 1 epoca | estendere fino a 1,5 solo se validation continua a migliorare |
| learning rate massimo | `3e-4` | da confermare con un breve run di stabilita' |
| warmup | 8.000 update | crescita lineare da zero |
| scheduler | cosine decay | termina a `3e-5` |
| AdamW beta | `0.9`, `0.95` | configurabile e scritto nel checkpoint |
| weight decay | `0.1` | non applicato ai pesi RMSNorm |
| clipping | norma globale `1.0` | misurare e registrare le occorrenze |
| validation | ogni 2.000 update | batch e seed fissi, separati dal test |
| checkpoint | ogni 2.000 update | mantenere anche il migliore per validation |

Il numero di token, non soltanto il numero di step, e' la metrica primaria del
run. Il log JSONL persistente include loss train, learning rate, norma del
gradiente, token/s, durata dello step e memoria Metal attiva/picco. I checkpoint
v5 sono scritti tramite file temporaneo, `fsync` e rename atomico e includono
uno SHA-256 dell'header e del payload; i checkpoint v1--v4 restano leggibili per
riprendere gli esperimenti esistenti. Gli eventi periodici aggiungono loss/PPL
validation e registrano i nuovi checkpoint best.

## Dati e allineamento conversazionale

Il pretraining usa il tokenizer train-only `italiano-v3.llmtok` e gli split
documentali del corpus multi-sorgente `italiano-v3`. Un tokenizer diverso puo'
avere lo stesso numero di ID ma assegnare loro byte completamente differenti:
in quel caso loss e training restano validi, ma la generazione decodificata
diventa illeggibile. Le diagnostiche confrontano percio' lo SHA-256 del
tokenizer con quello registrato nell'header `.llmdat`, non soltanto la dimensione
del vocabolario.
Un corpus generale aggiuntivo puo' essere introdotto solo con manifest che
registri origine, licenza, filtri, deduplicazione, identificatori e split.
Il trainer del tokenizer rifiuta manifest che non dichiarano esplicitamente
`tokenizer_input_split: train`, evitando contaminazione di validation e test.

Il fine-tuning chat e' una fase successiva e separata, ora implementata e
specificata in [Italiano-Chat-75M](italiano-chat-75m.md):

```text
Italiano-Base-75M
      -> dataset italiano istruzione/risposta con licenza verificata
      -> token di ruolo espliciti: <|user|>, <|assistant|>, <|end|>
      -> supervised fine-tuning con loss solo sulle risposte assistant
      -> Italiano-Chat-75M
```

Non usare conversazioni personali o dati privati nel dataset. Il formato
`.llmsft` conserva una mask per posizione e il trainer calcola loss e gradienti
soltanto sul contenuto assistant e sul relativo token di fine turno.

## Valutazione e criteri di uscita

Per dichiarare completato `Italiano-Base-75M` servono tutti i punti seguenti:

1. CPU e Metal concordano entro le tolleranze dichiarate su un decoder con
   almeno due layer, quattro head e SwiGLU, inclusi gradienti e AdamW.
2. Il modello canonico si crea su Metal e completa almeno uno step con
   micro-batch 4 e contesto 512 senza fallback o round-trip host intermedi.
3. Test unitari coprono divisibilita' delle head, causalita', piu' layer,
   gradient check su una configurazione piccola e ripresa del checkpoint.
4. Il training log mostra loss validation in miglioramento; il checkpoint
   selezionato e' quello con migliore validation, non quello finale per step.
5. Una suite versionata contiene almeno 100 prompt di completamento italiano,
   con controlli manuali su italiano, ripetizione e coerenza breve.
6. Il test split non viene usato per scegliere iperparametri o checkpoint.

I checkpoint 75M sono superiori al limite ragionevole per file Git ordinari.
Il repository deve versionare configurazione, manifest, checksum, comando,
commit e metriche; il file binario viene pubblicato attraverso un artefact
store o una release con checksum verificabile.
