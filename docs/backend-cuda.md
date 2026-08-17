# Backend CUDA — specifica

**Stato:** implementato e compilabile su host con toolkit CUDA; la prima
esecuzione di validazione su GPU reale non e' ancora stata fatta. Il backend
riempie l'intera `llm_backend_ops` e non richiede modifiche a modello, trainer,
dataset o checkpoint.

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
| `matmul_f32`, `matmul_ex_f32` | `cublasSgemm` |
| elementwise, `accumulate`, `silu`, `adamw` | kernel grid-stride |
| riduzioni, `softmax`, `rms_norm`, `cross_entropy` | kernel un blocco per riga, riduzione con `__shfl_xor_sync` |
| `rope`, `gather_rows`, `scatter_add_rows` | kernel grid-stride, scatter con `atomicAdd` |
| `attention_forward/backward` | kernel un blocco per riga di query, porting diretto della versione Metal |

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

Il costo di questa garanzia e' una lettura in piu' di ogni tensore prodotto. E'
il prezzo della parita' con gli altri due backend ed e' il primo candidato alla
misura quando il profilo sara' disponibile.

## Batch

`llm_backend_begin_batch` e `llm_backend_end_batch` scelgono il backend
dell'acceleratore attivo. Il trainer le usa al posto delle chiamate Metal
esplicite che aveva prima: senza batch il backend sincronizza dopo ogni
operazione e su GPU discreta questo azzera il beneficio dell'esecuzione
asincrona.

## Precisione

Il contratto runtime v1 e' F32 stretto. TF32 mantiene storage e accumulo in F32
e tronca solo la mantissa dei moltiplicandi: sui tensor core e' un guadagno
grande, ma cambia gli ultimi bit e i test di parita' confrontano con il backend
CPU. Per questo resta opt-in:

```sh
LLM_LAB_CUDA_TF32=1 ./build/release/llm-lab model train ...
```

Senza la variabile il backend imposta `CUBLAS_PEDANTIC_MATH`. BF16 vero
richiederebbe aprire `LLM_DTYPE_BF16` nel contratto ed e' un lavoro separato.

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

## Lavoro successivo

Da fare dopo la prima esecuzione su GPU reale, guidato dalla misura e non
dall'intuizione:

1. attention con GEMM batched cuBLAS piu' un kernel di softmax mascherato, al
   posto del kernel diretto;
2. valutazione del costo effettivo della scansione di finitezza per operazione;
3. vettorizzazione `float4` sui kernel elementwise, che sono bandwidth-bound;
4. metriche CUDA per step nel log JSONL di training.
