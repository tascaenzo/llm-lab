# Backend CUDA — specifica

**Stato:** implementato, compilabile su host con toolkit CUDA e verificato su
una RTX 4090 reale. Il backend riempie l'intera `llm_backend_ops` e non
richiede modifiche a modello, trainer, dataset o checkpoint.

Il backend CUDA esiste per una ragione operativa precisa: l'addestramento lungo
di [Italiano-Base-75M](italiano-base-75m.md) sui 2,5 miliardi di token dello
split train di `italiano-v3` non e' praticabile sul Mac mini M4 in tempi utili.
Il checkpoint prodotto localmente si riapre in cloud senza conversioni, perche'
il formato `.llmckpt` e gli artefatti `.llmdat` sono little-endian espliciti
byte per byte.

## Principio: niente riscrittura di cio' che NVIDIA fornisce gia'

Il progetto non usa framework di deep learning, ma cuBLAS non e' un framework:
e' la BLAS del vendor, l'esatto analogo di cio' che MPS e' gia' nel
[backend Metal](backend-metal.md). Tutti i prodotti matrice-matrice passano da
cuBLAS. I kernel scritti a mano coprono soltanto le operazioni per cui NVIDIA
non offre un equivalente diretto.

| Operazione | Implementazione |
|---|---|
| `matmul_f32`, `matmul_ex_f32` | `cublasGemmEx`, calcolo F32/TF32/BF16 selezionabile |
| elementwise, `accumulate`, `silu`, `adamw` | kernel grid-stride |
| riduzioni, `softmax`, `rms_norm`, `cross_entropy` | kernel un blocco per riga, riduzione con `__shfl_xor_sync` |
| `rope`, `gather_rows`, `scatter_add_rows` | kernel grid-stride, scatter con `atomicAdd` |
| `attention_forward/backward` | kernel un blocco per riga; softmax online nel forward e QK/dP fusi nel backward |

Il trainer riduce inoltre l'overhead comune ai backend: la norma globale usa un solo kernel
`accumulate_sum_squares` per parametro al posto di quattro operazioni, mentre AdamW azzera il
gradiente e valida NaN/Inf nello stesso passaggio. Sul modello 75M vengono eliminati 444 dispatch
per update; su CUDA spariscono anche le 111 scansioni post-AdamW dei parametri.

Nel profilo 75M la ripartizione dei FLOP giustifica questa scelta: la output
head su `V = 32.008` da sola vale circa un quarto del costo per token, e con
proiezioni e SwiGLU i GEMM coprono circa il 90% del totale. L'attention pesa
intorno al 9%, quindi il kernel diretto e' corretto per primo e ottimizzabile
dopo la misura, non prima.

## Modello di memoria

`allocate` restituisce l'indirizzo di un `llm_cuda_buffer`, mai il puntatore
device: la facade del runtime tratta lo storage come un valore di puntatore
host, e un indirizzo device verrebbe dereferenziato da codice host. E' la stessa
scelta gia' fatta da Metal, ed e' la ragione per cui il porting non ha richiesto
di toccare `src/runtime/memory.c`.

`copy` decide la direzione cercando i due estremi nella lista delle allocazioni:
device-device resta sullo stream, host-device e device-host attraversano il bus
PCIe e quindi chiudono prima un eventuale batch aperto. Il pool dei buffer
riusa le allocazioni liberate fino a 64 buffer e 1 GiB, perche' `cudaMalloc` e'
un'operazione sincronizzante e sul percorso caldo va evitata.

## Errori numerici e indici

Il contratto del runtime richiede che un valore non finito diventi
`LLM_NUMERICAL_ERROR` e che un indice fuori intervallo diventi
`LLM_INVALID_INDEX`. Su CPU e Metal la verifica e' una scansione host; su CUDA
sarebbe un trasferimento sul bus dopo ogni operazione. Il backend usa invece due
flag sticky in memoria device:

- ogni operazione accoda un kernel di scansione sul proprio output;
- fuori da un batch, `llm_cuda_finish` sincronizza, legge i flag e li traduce in
  uno stato;
- dentro un batch la lettura e' rimandata alla chiusura, esattamente come Metal
  rimanda il controllo quando `batch_active` e' attivo.

I kernel `gather` e `scatter_add` verificano comunque l'indice al proprio
interno: un indice fuori intervallo salta l'elemento invece di scrivere fuori
dalla tabella, quindi la memoria resta integra anche mentre l'esito della
scansione e' ancora in volo sullo stream.

`LLM_LAB_CUDA_NUMERICS=strict` mantiene questa garanzia per ogni output.
`LLM_LAB_CUDA_NUMERICS=step` evita invece le scansioni delle attivazioni transitorie, mantenendo
quelle che proteggono loss, indici e parametri master aggiornati. Questo riduce fortemente i kernel
accessori senza permettere che uno stato AdamW non finito venga salvato in silenzio.

## Batch

`llm_backend_begin_batch` e `llm_backend_end_batch` scelgono il backend
dell'acceleratore attivo. Il trainer le usa al posto delle chiamate Metal
esplicite che aveva prima: senza batch il backend sincronizza dopo ogni
operazione e su GPU discreta questo azzera il beneficio dell'esecuzione
asincrona.

## Precisione

Il contratto di storage resta F32. La variabile `LLM_LAB_CUDA_MATH` seleziona il compute type
passato a `cublasGemmEx`:

```sh
LLM_LAB_CUDA_MATH=f32          # CUBLAS_COMPUTE_32F_PEDANTIC
LLM_LAB_CUDA_MATH=tf32         # CUBLAS_COMPUTE_32F_FAST_TF32
LLM_LAB_CUDA_MATH=bf16-compute # CUBLAS_COMPUTE_32F_FAST_16BF
```

TF32 e BF16 compute cambiano gli ultimi bit dei prodotti, ma parametri, gradienti, momenti AdamW e
checkpoint restano FP32. `LLM_LAB_CUDA_TF32=0|1` continua a essere accettata come alias legacy.
BF16 di storage, che ridurrebbe anche memoria e banda delle attivazioni, richiederebbe invece
aprire `LLM_DTYPE_BF16` nel contratto ed e' un lavoro separato.

La pipeline RunPod riporta modalita' richiesta ed effettiva nei log, forza sempre `f32/strict` nel
preflight e scrive il training ottimizzato in un checkpoint candidato distinto.

## Build

Il backend viene compilato automaticamente quando CMake trova un toolchain CUDA,
il che accade sull'host cloud e mai sul Mac. Sugli host senza CUDA entra
`cuda_backend_stub.c` e `llm_backend_cuda_create` restituisce
`LLM_UNSUPPORTED_DEVICE`.

```sh
cmake --preset release
cmake --build --preset release
```

Per forzare o escludere esplicitamente:

```sh
cmake --preset release -DLLM_LAB_ENABLE_CUDA=OFF
```

Le architetture di default sono `70;80;89;90`, cioe' V100, A100, L4 o RTX 4090 e
H100. Non e' `native` di proposito: rilevare la GPU locale fallisce su una
macchina di build che non ne ha, che e' esattamente il caso del runner CI e di
un host di compilazione separato da quello di training. Quando la GPU di
destinazione e' nota conviene indicarla e basta, perche' il tempo di
compilazione scala con il numero di architetture:

```sh
cmake --preset release -DCMAKE_CUDA_ARCHITECTURES=80
```

`LLM_LAB_CUDA_DEVICE` sceglie l'indice della GPU quando ce n'e' piu' di una.
`LLM_LAB_BACKEND=cuda` nel `.env` rende CUDA il default di `model train`,
`model generate`, `model evaluate`, benchmark e profiler. Un `--backend`
esplicito prevale; se CUDA non e' disponibile il comando fallisce e non ripiega
silenziosamente sulla CPU.

## Verifica

`runtime.cuda_backend` esegue la suite di contratto condivisa con CPU e Metal,
piu' i controlli specifici del backend: le quattro combinazioni di trasposizione
di `matmul_ex` (il punto in cui l'ordinamento colonna-maggiore di cuBLAS si
sbaglia piu' facilmente), la propagazione di `LLM_NUMERICAL_ERROR` e
`LLM_INVALID_INDEX`, il batch, e la parita' dei logit di un decoder completo
contro il backend CPU con tolleranza `2e-3`.

Su una macchina senza GPU NVIDIA il test riporta il salto e passa, quindi la
suite resta verde sul Mac.

La verifica si divide in due parti con costi molto diversi. La compilazione non
richiede nessuna GPU: `nvcc` produce codice per un'architettura di destinazione,
non lo esegue. Il job CI `Ubuntu / CUDA build` sfrutta questo e compila il
backend a ogni push dentro l'immagine `nvidia/cuda:*-devel`, quindi gli errori
di tipo, di firma e di API cuBLAS si vedono dal Mac senza affittare niente. La
correttezza numerica invece richiede silicio vero, perche' dipende da barriere,
shuffle di warp e atomiche: li' serve una GPU, anche piccola.

```sh
ctest --preset release -R runtime.cuda_backend --output-on-failure
```

La parita' non e' e non puo' essere bit-exact: `scatter_add_rows` e i gradienti
di attention usano atomiche, quindi l'ordine di riduzione varia fra esecuzioni.
Vale gia' per Metal.

## Primo training CUDA reale — 20 agosto 2026

La prima sessione reale e' stata eseguita su una singola RTX 4090 RunPod. La
suite `runtime.cuda_backend` ha completato con successo sulla GPU fisica; il
preflight ha inoltre eseguito un update su una copia del checkpoint e ha
verificato che il file sorgente restasse immutato. Questo conferma sia il
contratto numerico del backend sia la ripresa sicura dei checkpoint Metal su
CUDA.

La sessione di training ha ripreso `Italiano-Base-75M` dallo step 86.044 e ha
concluso 1.000 update allo step 87.044. Ha salvato il checkpoint latest e un
nuovo best checkpoint, con validation loss `2.88134766` e perplexity
`17.83829688`.

| Misura | Metal, Mac mini M4 | CUDA, RTX 4090 | Confronto |
|---|---:|---:|---|
| Velocita' training osservata | ~0,64 step/s | ~5,72 step/s | ~8,9x piu' veloce su CUDA |
| Ultima validation registrata | loss 2,90314; PPL 18,23129 (step 63.044) | loss 2,88135; PPL 17,83830 (step 87.044) | miglioramento coerente, ma non e' un benchmark paritario: cambiano gli step del modello |
| Configurazione del modello | 75,01M, F32, contesto 512, accumulo 2 | identica | checkpoint e dataset sono portabili senza conversione |

Il confronto di velocita' usa la stessa configurazione canonica e rappresenta
il guadagno operativo utile per il run lungo. Le due validation non vanno invece
lette come confronto Metal contro CUDA: la seconda proviene da un checkpoint
addestrato per altri 24.000 update. Il suo scopo e' confermare che il passaggio
di backend non ha interrotto il miglioramento del modello.

Al termine di una sessione completata, l'entrypoint RunPod ora mantiene il
container inattivo invece di riavviare automaticamente lo stesso training. Il
Pod continua comunque a essere fatturato finche' non viene fermato
esplicitamente; checkpoint, log e dataset restano nel volume `/workspace`.

## Lavoro successivo

Da fare dopo la prima esecuzione su GPU reale, guidato dalla misura e non
dall'intuizione:

1. profilare una sessione lunga reale e registrare le metriche CUDA per step
   nel log JSONL, prima di cambiare i kernel;
2. eliminare le atomiche del backward attention con un workspace tiled, se il
   profilo CUDA mostra che restano dominanti;
3. valutazione del costo effettivo della scansione di finitezza per operazione;
4. vettorizzazione `float4` sui kernel elementwise, che sono bandwidth-bound.
