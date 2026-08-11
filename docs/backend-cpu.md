# Backend CPU — specifica implementativa

## 1. Scopo e stato

Questa specifica definisce l'architettura del backend CPU del runtime
tensoriale, la separazione dei sorgenti e il percorso per introdurre
parallelismo, SIMD e kernel ottimizzati senza contaminare le API del modello.

**Implementato:** backend sincrono configurabile, allocazione allineata,
thread pool persistente, `parallel_for`, kernel C FP32/U32, parallelizzazione
per intervalli o righe, matmul a blocchi e benchmark dedicato.

**Restano futuri:** tuning dei tile su piu' famiglie di CPU e integrazione BLAS
opzionale. I kernel elementwise e il micro-kernel multi-riga della matmul usano
NEON/FMA su ARM64, SSE2 su x86-64 e, con Clang/GCC, dispatch runtime verso
AVX2/FMA o AVX-512/FMA. Il fallback C portabile resta sempre disponibile.

La teoria introduttiva e' in
[Backend CPU: usare davvero il processore](../wiki/11-backend-cpu.md). Il
contratto generale dei tensori e delle operazioni rimane in
[runtime-tensoriale.md](runtime-tensoriale.md).

## 2. Obiettivi

Il backend deve:

- fornire una implementazione CPU completa del contratto del runtime;
- conservare kernel di riferimento leggibili e portabili;
- usare piu' core senza esporre thread nelle API tensoriali;
- poter selezionare kernel scalari, SIMD o di libreria a runtime/build time;
- mantenere ownership e ciclo di vita delle risorse espliciti;
- produrre risultati verificabili contro il percorso di riferimento;
- compilare con Clang, GCC e MSVC su macOS, Linux e Windows.

Non deve:

- inserire logica di layer, autograd o optimizer nel backend;
- creare thread per ogni chiamata numerica;
- dipendere obbligatoriamente da OpenMP, BLAS o librerie esterne;
- nascondere un fallback silenzioso per dtype o layout non supportati;
- usare benchmark temporali come test unitari pass/fail.

## 3. Organizzazione dei sorgenti

La struttura corrente e quella prevista sono:

```text
include/runtime/
  types.h
  backend.h
  tensor.h
  operations.h
  runtime.h              umbrella pubblico

src/runtime/
  backend.c
  backend_internal.h
  memory.c
  storage_internal.h
  operations.c
  tensor.c
  tensor_internal.h
  backends/
    cpu/
      cpu_backend.c
      cpu_internal.h
      cpu_kernels.c
      cpu_kernels.h
      cpu_operations.c     strategie e partizionamento per operazione
      cpu_simd.c           NEON, SSE2 e fallback portabile
      cpu_simd.h
      cpu_executor.c       thread pool e parallel_for
      cpu_executor.h
      cpu_atomic.h         atomiche C/Interlocked portabili
      cpu_features.c       rilevamento dei processori disponibili
      cpu_threads.c        astrazione pthread/Windows
      cpu_threads.h
    metal/                 backend GPU Apple separato
    cuda/                  futuro

tests/runtime/
  test_runtime.c
  test_operations.c
  test_language_operations.c
  test_cpu_backend.c       executor, configurazione e carichi paralleli

utils/benchmarks/runtime/
  benchmark_cpu.c          CLI e output tabellare/JSONL
  benchmark_suite.c        workload, statistiche e validazione

utils/benchmarks/
  compare_results.py       confronto con baseline della stessa macchina
  performance_suite.py     scenari rappresentativi CPU/Metal e confronto
```

I file comuni non includono header nativi specifici di un dispositivo.
Ogni directory sotto `backends/` contiene implementazione, header privati e
risorse del solo backend corrispondente. Non vengono create directory vuote per
backend ancora inesistenti.

### 3.1 Responsabilita' correnti

`cpu_backend.c` contiene:

- creazione e distruzione del contesto CPU;
- tabella `llm_backend_ops`;
- allocazione, copia, zero e fill della memoria host;
- interrogazione dei dtype supportati;
- sincronizzazione del backend;
- collegamento tra operazioni del runtime e kernel CPU.

`cpu_kernels.c` contiene:

- implementazioni numeriche C di riferimento;
- nessuna conoscenza di `llm_tensor`;
- nessuna validazione completa di forme e dispositivi;
- nessuna allocazione dinamica nel percorso numerico.

`cpu_operations.c` contiene soglie, job e callback che dividono il lavoro. In
questo modo il calcolo leggibile resta nei kernel e la politica di esecuzione
non viene ammassata nella tabella del backend.

`cpu_internal.h` e' privato alla libreria `runtime` e dichiara i contratti tra
backend, executor e kernel. Non puo' essere incluso dal modello o dagli utenti
dell'API pubblica.

## 4. Confine tra API, backend, executor e kernel

```text
llm_matmul(backend, A, B, C)
  |
  | operations.c: valida backend, dtype, shape, layout e aliasing
  v
backend->ops->matmul_f32(...)
  |
  | cpu_backend.c: sceglie strategia e implementazione
  v
cpu_executor_parallel_for(...)
  |
  | assegna intervalli o blocchi ai worker
  v
kernel scalare / SIMD / BLAS
```

La validazione che dipende dai tensori resta in `operations.c`. I kernel
ricevono puntatori, dimensioni e parametri gia' validati. Il backend sceglie la
strategia; l'executor non conosce la matematica dell'operazione.

## 5. Contesto del backend CPU

Oggi `llm_backend` conserva dispositivo e tabella delle operazioni. Per il
parallelismo dovra' possedere un contesto privato con almeno:

```c
typedef struct llm_cpu_context {
    llm_cpu_executor *executor;
    size_t thread_count;
    int deterministic;
} llm_cpu_context;
```

La struttura esatta puo' cambiare durante l'implementazione; le invarianti no:

- il contesto nasce con il backend e viene distrutto con esso;
- tutti i tensori del backend devono essere distrutti prima del backend;
- i worker vengono avviati una volta, non per operazione;
- la distruzione attende i lavori gia' accettati e termina tutti i worker;
- un errore di inizializzazione non lascia thread o memoria attivi;
- nessuna struttura del thread pool e' pubblica in `runtime.h`.

### 5.1 Configurazione pubblica

La creazione semplice deve continuare a funzionare:

```c
llm_backend_cpu_create(&backend);
```

Una seconda API configurabile verra' aggiunta solo insieme all'executor:

```c
typedef struct llm_cpu_backend_config {
    size_t thread_count; /* 0 = scelta automatica */
    int deterministic;
} llm_cpu_backend_config;

llm_status llm_backend_cpu_create_with_config(
    const llm_cpu_backend_config *config,
    llm_backend **out_backend);
```

Questa API e' dichiarata in `backend.h`, incluso anche dall'umbrella
`runtime.h`. La funzione semplice equivale a
`thread_count = 0` e modalita' deterministica attiva. La funzione
`llm_backend_cpu_thread_count()` permette di leggere il numero effettivo scelto.

La scelta automatica usa i processori logici disponibili ma applica un limite
ragionevole e permette un override esplicito. Il backend deve funzionare anche
con `thread_count = 1`, modalita' fondamentale per test e debugging.

## 6. Executor CPU

L'executor e' un thread pool portabile con una primitiva interna concettuale:

```c
typedef llm_status (*llm_cpu_range_fn)(void *context,
                                      size_t begin,
                                      size_t end);

llm_status llm_cpu_parallel_for(
    llm_cpu_executor *executor,
    size_t item_count,
    size_t minimum_items_per_task,
    llm_cpu_range_fn function,
    void *context);
```

La firma e' implementata nell'header privato e mantiene questi comportamenti:

1. `[0, item_count)` viene diviso in intervalli disgiunti;
2. ogni elemento viene elaborato esattamente una volta;
3. il thread chiamante partecipa all'esecuzione;
4. sotto una soglia il lavoro viene eseguito direttamente;
5. la chiamata ritorna solo quando tutti gli intervalli sono completati;
6. nessun worker conserva puntatori al contesto del job dopo il ritorno;
7. non sono permesse chiamate annidate che causino deadlock.

La v1 resta sincrona dal punto di vista dell'API pubblica. Di conseguenza
`llm_backend_synchronize()` rimane un no-op dopo che ogni operazione e'
ritornata. L'asincronia tra operazioni e' un problema separato e futuro.

### 6.1 Portabilita' dei thread

`cpu_threads.c` incapsula pthread su macOS/Linux e thread, critical section e
condition variable Win32 su Windows. Questa scelta evita di dipendere da
`<threads.h>`, che non e' disponibile in tutte le toolchain C23 supportate. Le
API di sistema non escono da `backends/cpu/`.

### 6.2 Scheduling

La politica corrente usa chunk dinamici con assegnazione atomica:

- numero di task limitato rispetto ai worker;
- intervalli contigui per preservare localita' di memoria;
- soglia specifica per famiglia di operazioni;
- nessun mutex globale per reclamare ogni chunk;
- mutex e condition variable usati soltanto per pubblicare e completare un job;
- nessun work stealing finche' un benchmark non ne dimostra la necessita'.

Le soglie non vengono scelte “a intuito” come costanti definitive. Partono da
valori conservativi, sono nominate e vengono misurate su matrici e tensori
rappresentativi.

## 7. Strategie per operazione

### 7.1 Elementwise

`add`, `multiply`, `scale`, `fill` e `zero` dividono l'intervallo lineare degli
elementi. Gli output non si sovrappongono, quindi non servono lock.

Per input piccoli si usa il kernel diretto. Per input grandi ogni task riceve
puntatori gia' spostati e una lunghezza.

### 7.2 Riduzioni

Le riduzioni correnti lavorano sull'ultima dimensione. La prima
parallelizzazione distribuisce le righe esterne: ogni worker scrive un elemento
di output distinto e riduce la singola riga in ordine scalare.

Non si divide inizialmente una singola riga tra worker. Tale strategia richiede
risultati parziali, un ordine di combinazione e workspace. Potra' essere aggiunta
per righe eccezionalmente larghe.

### 7.3 Matmul

Il kernel di riferimento resta il triplo ciclo corretto. L'evoluzione prevista
e':

1. scegliere un ordine dei cicli con accessi contigui;
2. dividere righe o tile di `C` tra worker;
3. introdurre tiling per cache L1/L2;
4. vettorizzare il micro-kernel;
5. confrontare il percorso interno con una BLAS opzionale.

Ogni task possiede una regione disgiunta di `C`. Non sono richiesti aggiornamenti
atomici. Le dimensioni dei tile devono essere parametri interni documentati e
misurati, non parte dell'API pubblica.

### 7.4 Gather

Gli indici vengono divisi in intervalli. Ogni indice produce una riga di output
distinta; le letture duplicate dalla tabella sono sicure.

### 7.5 Scatter-add

Indici duplicati possono provocare scritture concorrenti sulla stessa riga. Il
backend evita la race partizionando invece le colonne: ogni task possiede un
intervallo di colonne disgiunto e attraversa gli indici nello stesso ordine.
Non servono atomiche o buffer temporanei e il risultato resta deterministico.
La strategia e' conveniente quando le righe sono abbastanza larghe; le soglie
evitano di distribuire lavoro minuscolo.

### 7.6 Softmax e cross-entropy

Le righe sono indipendenti. Ogni worker esegue massimo, esponenziali e somma
della propria riga. La stabilizzazione tramite sottrazione del massimo e i
controlli numerici devono restare identici al riferimento.

La cross-entropy forward produce loss parziali. In modalita' deterministica
queste vengono combinate in un ordine fisso. Il backward scrive righe di
gradiente distinte.

## 8. SIMD e rilevamento delle capacita'

`cpu_features.c` rileva il numero di processori logici disponibili e applica il
limite interno. Il livello SIMD usa istruzioni baseline garantite
dall'architettura: NEON su ARM64 e SSE2 su x86-64. Add, multiply, scale e il
passo AXPY della matmul gestiscono anche la coda scalare.

Il micro-kernel della matmul aggiorna quattro righe di output insieme, riusando
il vettore letto dalla matrice destra. Su x86 con Clang/GCC il binario contiene
anche versioni AVX2/FMA e AVX-512/FMA compilate con target dedicato; il dispatch
controlla le feature della CPU prima di eseguirle. Su toolchain che non supportano
questo meccanismo resta il percorso SSE2.

Il dispatch rispetta questa precedenza:

```text
implementazione esplicitamente disabilitata
  -> kernel C portabile

feature compilata e disponibile sulla CPU
  -> kernel SIMD compatibile

feature non disponibile
  -> kernel C portabile
```

Un binario non deve eseguire istruzioni che la CPU corrente non supporta. I
kernel specifici possono vivere in file separati quando richiedono flag di
compilazione differenti. L'auto-vettorizzazione del compilatore e' accettata,
ma non sostituisce il dispatch esplicito quando vengono usate intrinsic.

Priorita' iniziale:

- ARM64/NEON per Apple Silicon e ARM64;
- x86-64 con baseline portabile e percorsi AVX2/FMA verificati;
- fallback scalare per tutte le piattaforme.

## 9. BLAS opzionale e oversubscription

Una libreria BLAS puo' fornire una matmul molto ottimizzata, ma non deve essere
una dipendenza obbligatoria del progetto. L'integrazione deve essere controllata
da una opzione CMake e avere un percorso interno sempre disponibile.

Molte BLAS usano gia' thread propri. Il backend non deve parallelizzare
esternamente una chiamata BLAS multithread, altrimenti puo' creare piu' thread
del necessario. La politica deve essere una sola tra:

- BLAS gestisce il parallelismo della matmul;
- BLAS a un thread, backend distribuisce blocchi;
- kernel interno gestito dall'executor.

La scelta effettiva deve comparire nell'output del benchmark/configurazione.

## 10. Memoria, cache e workspace

Le allocazioni host restano allineate almeno a 64 byte. Questo facilita linee di
cache e caricamenti SIMD, senza garantire da solo prestazioni.

Il percorso numerico di riferimento non alloca. Se un kernel parallelo richiede
workspace, questo deve essere:

- posseduto dal contesto o passato esplicitamente;
- riutilizzato tra operazioni;
- dimensionato con controlli di overflow;
- separato per worker quando contiene dati mutabili;
- liberato alla distruzione del backend.

False sharing va evitato per contatori, risultati parziali e strutture dei
worker, usando separazione o padding dove una misura lo giustifica.

## 11. Errori e ciclo di vita

La creazione configurabile deve validare tutti i parametri prima di pubblicare
il backend. In caso di errore `*out_backend` resta `NULL`.

Errori interni del thread pool diventano `LLM_BACKEND_ERROR`; errori di memoria
restano `LLM_ALLOCATION_FAILED`. Un kernel avviato deve completare o propagare
uno stato definito prima del ritorno. La v1 dell'executor non supporta
cancellazione parziale dei job.

`llm_backend_destroy()` dovra' delegare una distruzione specifica al backend
prima di liberare la struttura generica. Questa modifica e' obbligatoria quando
`llm_backend` iniziera' a possedere il contesto CPU.

## 12. Correttezza numerica e determinismo

I kernel ottimizzati vengono confrontati con il kernel di riferimento. Le
tolleranze devono dipendere dall'operazione e dalla scala dei valori.

La modalita' deterministica richiede:

- partizionamento stabile a parita' di configurazione;
- ordine stabile di combinazione delle riduzioni;
- nessuna race su output o workspace;
- risultati ripetibili sulla stessa piattaforma e build.

Non promette identita' bit per bit tra architetture, compilatori o diverse
implementazioni SIMD. Tale promessa sarebbe incompatibile con molte
ottimizzazioni lecite in virgola mobile.

## 13. Test

### 13.1 Test unitari dell'executor

- zero, uno e molti elementi;
- lavoro inferiore e superiore alla soglia;
- ogni indice visitato esattamente una volta;
- uso con uno e piu' worker;
- creazione e distruzione ripetute;
- propagazione degli errori di inizializzazione;
- assenza di deadlock sotto ThreadSanitizer, dove disponibile.

### 13.2 Test di conformita' dei kernel

Per ogni operazione parallelizzata:

- confronto 1 thread contro il riferimento;
- confronto N thread contro il riferimento;
- dimensioni zero invalide e forme limite valide;
- dimensioni non multiple del chunk o della larghezza SIMD;
- input casuali con seed fisso;
- `NaN`/`Inf` secondo il contratto esistente;
- ripetizione in modalita' deterministica;
- indici duplicati per scatter-add.

### 13.3 Sanitizer

AddressSanitizer e UndefinedBehaviorSanitizer restano obbligatori nelle
configurazioni supportate. ThreadSanitizer viene aggiunto per executor e kernel
paralleli su una toolchain compatibile.

## 14. Benchmark

I benchmark sono eseguibili separati dai test. Devono riportare almeno:

- nome operazione e implementazione scelta;
- forma e dtype;
- numero di thread;
- iterazioni di warm-up e misurate;
- mediana o minimo robusto della durata;
- throughput appropriato: elementi/s, GB/s o GFLOP/s;
- informazioni essenziali su CPU e build.

La suite implementata misura tutti i 29 workload del contratto CPU:

- memoria: zero, fill, copy e cast nelle due direzioni;
- elementwise: add, multiply, scale e accumulate;
- riduzioni: sum, max e mean-square;
- matmul normale, mixed precision e con transpose logica sinistra/destra;
- gather e scatter-add;
- SiLU, RMSNorm, RoPE e attention GQA causale, forward e backward;
- softmax e cross-entropy forward/backward;
- AdamW;
- ogni lista richiesta di thread, inclusa la scelta automatica.

Per ogni caso registra minimo, mediana, P95, media, deviazione standard,
coefficiente di variazione, nanosecondi per chiamata, chiamate al secondo,
throughput specifico dell'operazione e un valore di controllo numerico.

Il benchmark deve impedire che il compilatore elimini il calcolo e deve
riutilizzare input/output tra iterazioni. Non vengono fissate soglie assolute in
CI: i risultati servono per decisioni e regressioni osservate in ambiente
controllato.

La suite si costruisce ed esegue cosi'. Il formato predefinito e' una tabella
pensata per la lettura diretta:

```sh
cmake --preset release -DLLM_LAB_BUILD_BENCHMARKS=ON
cmake --build --preset release --target runtime_benchmark
./build/release/utils/benchmarks/runtime_benchmark \
  --backend cpu \
  --operations all \
  --threads 1,2,4,8,auto \
  --elements 1048576 \
  --rows 256 --columns 256 --inner 256 \
  --batch 1 --sequence 128 --query-heads 8 --kv-heads 2 --head-dim 64 \
  --warmup 3 --iterations 20 --sample-ms 10
```

Il target unico `runtime_benchmark` usa `--backend cpu|metal|all` per forzare il
backend e `--precision f32|f16|bf16` per scegliere la precisione. CUDA verra'
aggiunto allo stesso selettore quando il relativo backend sara' disponibile.

La tabella mostra mediana, p95, throughput, speedup ed efficienza rispetto alla
prima configurazione della lista. I campioni troppo brevi vengono ripetuti
automaticamente per raggiungere la durata minima configurata. Il risultato
numerico viene verificato dopo le misure.

Per automazione, archiviazione e confronti si usa `--format jsonl`. In questa
modalita' ogni riga di stdout e' un record JSON che include anche minimo, media,
deviazione standard, coefficiente di variazione e metadati dell'ambiente.

Per creare e confrontare una baseline sulla **stessa macchina**, con lo stesso
carico di sistema e la stessa build:

```sh
./build/release/utils/benchmarks/runtime_benchmark --backend cpu \
  --format jsonl \
  --operations all --threads 1,2,4,8,auto \
  > baseline.jsonl

./build/release/utils/benchmarks/runtime_benchmark --backend cpu \
  --format jsonl \
  --operations all --threads 1,2,4,8,auto \
  > current.jsonl

python3 utils/benchmarks/compare_results.py \
  baseline.jsonl current.jsonl \
  --max-regression-percent 10
```

In alternativa `performance_suite.py` esegue automaticamente scenari piccoli,
memory-bound, matriciali e Transformer, salva i record e applica la soglia:

```sh
python3 utils/benchmarks/performance_suite.py \
  --output artifacts/benchmarks/local/baseline.jsonl

python3 utils/benchmarks/performance_suite.py \
  --baseline artifacts/benchmarks/local/baseline.jsonl \
  --output artifacts/benchmarks/local/current.jsonl \
  --max-regression-percent 10
```

Le soglie temporali non vengono applicate nella CI condivisa, dove hardware e
rumore non sono controllabili. La CI compila il benchmark ed esegue soltanto uno
smoke test funzionale; il confronto prestazionale e' destinato a runner stabili.

## 15. Sequenza di implementazione

### Incremento CPU-A — Separazione strutturale

**Stato: implementato.**

- directory `src/runtime/backends/cpu/`;
- separazione tra gestione backend e kernel;
- tabella delle capacita' dtype posseduta dal backend CPU;
- build e test invariati nel comportamento pubblico.

### Incremento CPU-B — Executor portabile

**Stato: implementato.**

- contesto del backend e distruttore specifico;
- configurazione del numero di thread;
- thread pool persistente;
- `parallel_for` sincrono;
- test isolati dell'executor.

**Uscita:** lo stesso job produce risultato corretto con 1 e N thread senza
leak, race o deadlock.

### Incremento CPU-C — Parallelizzazione dei kernel

**Stato: implementato.**

- elementwise, righe delle riduzioni, gather, softmax e cross-entropy;
- matmul inizialmente divisa per righe o blocchi di output;
- scatter-add partizionato per colonne senza scritture condivise;
- soglie per evitare overhead sui tensori piccoli.

Anche la cross-entropy forward resta seriale per mantenere una riduzione della
loss semplice e deterministica. Il backward e' parallelo per riga.

**Uscita:** tutti i test di conformita' passano con 1 e N thread.

### Incremento CPU-D — Misurazione e matmul a blocchi

**Stato: implementato nella versione iniziale.**

- benchmark riproducibile;
- ordine dei cicli e tiling della matmul;
- misure di scalabilita' per numero di thread;
- documentazione dei risultati e delle scelte.

Il target unico `runtime_benchmark` misura tutte le primitive principali con
backend e configurazioni di thread espliciti e produce JSONL confrontabile con
una baseline. La matmul usa tile interni e per colonna da 64
elementi; il tuning per architettura resta futuro.

**Uscita:** miglioramento misurato su forme rappresentative senza regressioni di
correttezza.

### Incremento CPU-E — SIMD e BLAS opzionale

**Stato: SIMD e dispatch x86 implementati; BLAS futura.**

- rilevamento feature;
- percorsi NEON/SSE2 e micro-kernel multi-riga;
- dispatch AVX2/FMA e AVX-512/FMA su Clang/GCC x86;
- code path portabile sempre disponibile;
- BLAS opzionale e politica esplicita dei thread;
- test forzabili per ciascun dispatch disponibile.

**Uscita:** il backend seleziona soltanto implementazioni supportate e ogni
percorso coincide numericamente col riferimento.

### Incremento CPU-F — Primitive di training Transformer

**Stato: implementato come riferimento numerico F32.**

- `cpu_linear.c`: matmul con transpose logica e accumulo in-place;
- `cpu_transformer.c`: SiLU, RMSNorm, RoPE e attention GQA causale, con forward
  e backward;
- `cpu_optimizer.c`: aggiornamento AdamW con correzione del bias;
- validazione comune nell'API pubblica, senza concetti di modello nei kernel;
- gradient check numerici per SiLU, RMSNorm e attention;
- test con AddressSanitizer e UndefinedBehaviorSanitizer.

Queste implementazioni privilegiano leggibilita', determinismo e correttezza.
In particolare attention backward e RMSNorm backward sono inizialmente seriali
per evitare accumuli concorrenti sui gradienti condivisi. La parallelizzazione
e le fusioni verranno guidate dai benchmark dopo la parita' Metal.

## 16. Criterio per procedere ai layer neurali

Il backend CPU non deve essere “perfetto”, ma prima dei layer neurali devono essere
veri questi punti:

- struttura per backend stabile e comprensibile;
- kernel di riferimento conservati;
- executor configurabile funzionante con 1 e N thread;
- operazioni principali parallelizzate senza race;
- almeno una matmul parallela misurata;
- test, sanitizer e benchmark documentati;
- forward, backward e AdamW del contratto F32 disponibili;
- nessun dettaglio di thread visibile nel futuro codice dei layer.

SIMD completo, BLAS, kernel fusi e tuning per ogni CPU possono proseguire anche
dopo l'inizio del core neurale.
