# Backend Metal — specifica implementativa

## 1. Obiettivo e stato

Il backend Metal esegue il contratto del runtime tensoriale sulla GPU Apple senza
esporre tipi Objective-C o Metal al codice del modello. Le API comuni continuano
a ricevere `llm_backend`, `llm_tensor` e `llm_status`.

**Stato:** backend operativo e verificato su Apple Silicon. Implementa tutte le
primitive del runtime, batch asincroni espliciti, pool dei buffer, metriche GPU,
riduzioni adattive, kernel elementwise vettoriali, tre famiglie di matmul FP32 e
calcolo misto FP16/BF16 con accumulo FP32. La selezione della matmul viene
autotarata per device, dtype e forma. La suite benchmark misura CPU e Metal con
lo stesso workload.

Non e' ancora una libreria GPU generale: mancano autograd, kernel dei layer
Transformer, fusion estesa e tuning persistente su disco. Questi sono incrementi
separati dal backend numerico corrente.

La spiegazione introduttiva e' in
[Backend Metal: portare i tensori sulla GPU Apple](../wiki/12-backend-metal.md).

## 2. I dieci incrementi implementati

1. Device, command queue e compilazione delle compute pipeline.
2. Buffer `MTLStorageModeShared` con ownership e controlli di device.
3. Pool con riuso best-fit, limite di 32 buffer e 256 MiB in cache.
4. Batch asincrono esplicito e sincronizzazione su fine batch o accesso host.
5. Metriche per startup, command buffer, dispatch, tempo GPU e riuso memoria.
6. Riduzioni e softmax parallele tramite threadgroup e primitive SIMD-group.
7. Cross-entropy forward/backward parallela sulla dimensione del vocabolario.
8. Scatter-add concorrente con aggiornamento atomico FP32 tramite CAS.
9. Matmul FP32 tiled 16 e 32, con selezione automatica in base alla forma.
10. FP16/BF16, cast e matmul mista tiled con accumulo FP32, test e benchmark.

## 3. API pubblica Metal

```c
int llm_backend_metal_is_available(void);
llm_status llm_backend_metal_create(llm_backend **out_backend);
const char *llm_backend_metal_device_name(const llm_backend *backend);

llm_status llm_backend_metal_begin_batch(llm_backend *backend);
llm_status llm_backend_metal_end_batch(llm_backend *backend);

llm_status llm_backend_metal_get_metrics(
    const llm_backend *backend,
    llm_metal_backend_metrics *out_metrics);
llm_status llm_backend_metal_reset_metrics(llm_backend *backend);
```

La factory restituisce `LLM_UNSUPPORTED_DEVICE` quando il sistema non espone un
device Metal. Il nome del device appartiene al backend e resta valido fino alla
sua distruzione.

Le API comuni aggiunte per la precisione ridotta sono:

```c
llm_status llm_cast(llm_backend *backend,
                    const llm_tensor *input,
                    llm_tensor *output);

llm_status llm_matmul_mixed_f32(llm_backend *backend,
                                const llm_tensor *left,
                                const llm_tensor *right,
                                llm_tensor *output);
```

`llm_cast` accetta `F32 <-> F16` e `F32 <-> BF16` a parita' di forma.
`llm_matmul_mixed_f32` richiede input dello stesso tipo F16 o BF16 e produce un
output F32. L'accumulatore FP32 limita la perdita numerica durante la somma.

## 4. Organizzazione del codice

```text
src/runtime/backends/metal/
  metal_backend.m       device, queue, pipeline, batch e metriche
  metal_memory.m        pool, registro buffer, copie e sincronizzazione host
  metal_operations.m    encoding, selezione e dispatch
  metal_internal.h      tipi e contratti privati
  metal_backend_stub.c  comportamento non-Apple
  kernels/
    runtime.metal       kernel Metal Shading Language

tests/runtime/
  test_metal_backend.c  lifecycle, batch, memoria, kernel e parita' CPU

utils/benchmarks/runtime/
  benchmark_cpu.c       CLI unificata CPU/Metal
  benchmark_suite.c     workload, verifica e statistiche
```

I file `.m` sono il solo ponte Objective-C. Header pubblico, validazione comune
e chiamanti rimangono C23.

## 5. Build e pipeline

Su Apple, CMake abilita Objective-C e collega `Foundation` e `Metal`. Se la
toolchain espone `metal` e `metallib`, CMake compila `runtime.metal` offline e
incorpora la `.metallib` nel binario. Con i soli Command Line Tools, che possono
non includere questi programmi, incorpora anche il sorgente e usa
`newLibraryWithSource` come fallback automatico. Le pipeline vengono create una
sola volta per backend e `pipeline_compilation_seconds` rende visibile il costo
di startup.

Linux e Windows compilano `metal_backend_stub.c`: le stesse API esistono, ma il
device risulta non disponibile. In questo modo il progetto e la CI restano
portabili.

## 6. Memoria e pool

Ogni storage Metal conserva un wrapper opaco con `id<MTLBuffer>`, dimensione
logica e capacita'. Il contesto mantiene due liste protette da mutex:

- buffer attivi, posseduti dai tensori;
- buffer in cache, disponibili per il riuso.

Alla creazione di un tensore viene scelto il buffer piu' piccolo che possa
contenere la richiesta. Alla distruzione il buffer entra nel pool finche' non
vengono superati 32 elementi o 256 MiB; oltre il limite viene rilasciato.

Le copie distinguono host-buffer, buffer-host e buffer-buffer. Una lettura o
scrittura host completa prima eventuali comandi pendenti, evitando accessi
concorrenti alla memoria condivisa.

## 7. Esecuzione sincrona e batch

Fuori da un batch ogni operazione invia un command buffer e ne attende il
completamento. E' la modalita' semplice e compatibile con il backend CPU.

Tra `llm_backend_metal_begin_batch` e `llm_backend_metal_end_batch`, invece, le
operazioni compute condividono lo stesso encoder e lo stesso command buffer.
Un blit chiude l'encoder compute e continua sul medesimo command buffer.
`end_batch` invia una sola volta il lavoro, attende e propaga gli errori.

```c
llm_backend_metal_begin_batch(backend);
llm_add(backend, &a, &b, &temporary);
llm_scale(backend, &temporary, 0.5f, &output);
llm_backend_metal_end_batch(backend);
```

Un accesso host forza il completamento del lavoro pendente. I batch annidati e
la chiusura di un batch inesistente restituiscono `LLM_INVALID_ARGUMENT`.

## 8. Kernel paralleli

| Famiglia | Strategia |
|---|---|
| fill/add/multiply/scale/cast | un thread per quattro elementi, con coda sicura |
| sum/max/mean-square | un threadgroup per riga, riduzione SIMD-group |
| softmax | massimo, esponenziale e somma paralleli per riga |
| cross-entropy forward | un threadgroup per riga e accumulo atomico della loss |
| cross-entropy backward | un threadgroup per riga |
| gather | un thread per elemento di output |
| scatter-add | un thread per elemento e CAS atomico sul float |
| matmul piccola | tile 16 x 16, un output per thread |
| matmul grande | tile 32 x 32, quattro output per thread |
| matmul SIMD-group | matrici SIMD 8 x 8 su GPU Apple family 7 o successiva |

Le riduzioni usano `simd_sum` e `simd_max`, poi combinano i risultati dei gruppi
SIMD in memoria threadgroup. Il backend adatta 32, 64, 128 o 256 thread alla
larghezza della riga, rispettando `threadExecutionWidth` e
`maxTotalThreadsPerThreadgroup`.

Lo scatter-add rappresenta il float come bit `uint` e ripete un compare-and-swap
finche' l'aggiornamento riesce. Indici duplicati sono quindi corretti ma l'ordine
delle somme concorrenti non e' bit-deterministico.

## 9. Matmul e precisione

La variante 16 x 16 limita il lavoro sprecato sulle forme piccole. La variante
32 x 32 usa 256 thread: ogni thread calcola quattro elementi del risultato e i
dati caricati nel tile vengono riutilizzati piu' volte. Sulle GPU compatibili,
la terza variante FP32 usa `simdgroup_matrix` 8 x 8 ed e' ammessa quando M, K e N
sono multipli di 16.

Le varianti F16 conservano tile `half` in memoria threadgroup. Le varianti BF16
caricano valori a 16 bit e li convertono in FP32, perche' MSL non offre un tipo
BF16 portabile in tutte le versioni supportate. Entrambe accumulano in FP32.

Alla prima matmul sincrona di una combinazione `(dtype, M, K, N)`, il backend
esegue un warm-up per variante, intercala tre prove per non favorire il kernel
misurato per ultimo e confronta la mediana dei tempi GPU. La variante piu'
rapida viene conservata nel contesto e riusata. Il warm-up del benchmark assorbe
questo costo. Durante un batch con cache ancora vuota viene usata una scelta
conservativa, senza forzare una sincronizzazione. Per diagnosi riproducibili si puo' impostare
`LLM_METAL_MATMUL_PIPELINE=tile16`, `tile32` o `simdgroup`; una variante non
compatibile con dtype o forma viene ignorata.

## 10. Metriche

`llm_metal_backend_metrics` espone:

- buffer attivi, buffer/byte in cache e allocazioni riusate;
- command buffer inviati e kernel dispatchati;
- tempo di compilazione delle pipeline;
- ultimo tempo GPU e tempo GPU cumulativo.

I tempi GPU derivano dai timestamp del command buffer. Il tempo end-to-end del
benchmark include anche encoding, invio, sincronizzazione e controlli host; per
questo e' sempre il dato principale per confrontare backend reali.

## 11. Benchmark riproducibile

Configurazione Release:

```sh
cmake --preset release -DLLM_LAB_BUILD_BENCHMARKS=ON
cmake --build --preset release --target runtime_benchmark
```

Confronto CPU/Metal FP32:

```sh
./build/release/utils/benchmarks/runtime_benchmark \
  --backend all \
  --operations matmul \
  --threads 1,2,4,8,auto \
  --precision f32 \
  --rows 512 --inner 512 --columns 512
```

Misura FP16 o BF16:

```sh
./build/release/utils/benchmarks/runtime_benchmark \
  --backend metal \
  --operations matmul \
  --precision f16 \
  --rows 512 --inner 512 --columns 512
```

`--format jsonl` produce record schema 3 con backend, device, dtype, statistiche
end-to-end, tempi GPU e startup. Lo stesso eseguibile forza CPU, Metal oppure
entrambi tramite `--backend`, senza suite separate per hardware.

### Report automatico della macchina

`runtime_benchmark_report` rileva hostname, sistema operativo, architettura,
modello CPU, thread logici, memoria e device Metal. Esegue tutte le primitive
supportate sulla CPU e, quando disponibile, su Metal. La matmul viene misurata
in FP32, FP16 e BF16.

```sh
cmake --build --preset release --target runtime_benchmark_report
./build/release/utils/benchmarks/runtime_benchmark_report
```

Profili disponibili:

| Profilo | Uso |
|---|---|
| `--quick` | smoke test rapido e verifica funzionale |
| `--standard` | report bilanciato, usato quando non si passa un'opzione |
| `--full` | forme maggiori e campioni piu' lunghi |

L'output mostra prima l'hardware rilevato, poi una tabella separata per ogni
dispositivo. Ogni tabella riporta forma, mediana, p95, tempo GPU e velocita'. Il
riepilogo indica kernel riusciti/falliti, velocita' delle matmul e confronto
Metal/CPU per FP32.

### Suite prestazionale rappresentativa

Il report hardware serve per esplorare la macchina. Per decidere se una modifica
ha migliorato o peggiorato il runtime si usa invece `performance_suite.py`.
La configurazione predefinita esegue tutti i kernel CPU e tutti quelli Metal
disponibili. La durata dipende soprattutto dalle forme di attention e matmul e usa:

- 5 iterazioni di warm-up;
- 40 campioni misurati;
- almeno 30 ms per campione;
- CPU con thread automatici e Metal quando disponibile;
- operazioni piccole per misurare la latenza;
- vettori grandi per la banda di memoria;
- righe `512 x 2048` per riduzioni, gather, scatter-add, softmax e cross-entropy;
- RoPE e attention GQA, forward e backward, su forme Transformer dedicate;
- SiLU, RMSNorm e AdamW nei percorsi CPU di training;
- matmul `512 x 512 x 512` in FP32, FP16 e BF16;
- matmul Transformer `512 x 1024 x 4096` in FP32.

Durante l'esecuzione ogni risultato completato aggiorna una barra globale e
mostra backend, kernel, tipo, mediana e throughput. Al termine viene stampato un
riepilogo compatto con hardware rilevato, test CPU/Metal completati, durata,
stabilita' dei campioni e confronto delle matmul principali. Quando e' presente
una baseline, il riepilogo conta anche risultati migliorati, stabili, piu' lenti,
regressioni e casi mancanti.

Esecuzione diretta senza salvare una baseline:

```sh
cmake --build --preset release --target runtime_performance_suite
```

Prima di un'ottimizzazione si registra la baseline locale:

```sh
python3 utils/benchmarks/performance_suite.py \
  --output artifacts/benchmarks/local/baseline.jsonl
```

Dopo la modifica si ripete la stessa suite sulla stessa macchina:

```sh
python3 utils/benchmarks/performance_suite.py \
  --baseline artifacts/benchmarks/local/baseline.jsonl \
  --output artifacts/benchmarks/local/current.jsonl
```

Il confronto classifica ogni caso come:

- `MIGLIORATO` oltre il 3% piu' veloce;
- `STABILE` entro il rumore del 3%;
- `PIU' LENTO` oltre il 3% ma entro la soglia accettata;
- `REGRESSIONE` oltre il 10%, con codice di uscita non zero.

Le percentuali sono configurabili con `--noise-percent` e
`--max-regression-percent`. Baseline e risultati locali sono ignorati da Git,
perche' sono validi soltanto sulla macchina che li ha prodotti. La CI esegue uno
smoke test funzionale breve, non applica soglie temporali su runner condivisi.

Misure locali indicative su Apple M4, build Release, matmul 512 x 512 x 512,
5 warm-up e 40 campioni:

| Input | Mediana end-to-end | Mediana GPU | Throughput end-to-end |
|---|---:|---:|---:|
| FP32 | 0,515 ms | 0,207 ms | 520,8 GFLOP/s |
| FP16 | 0,504 ms | 0,196 ms | 532,9 GFLOP/s |
| BF16 | 0,567 ms | 0,252 ms | 473,2 GFLOP/s |

Sono misure di una singola macchina, non garanzie. Frequenza, temperatura,
carico del sistema e versione del compilatore cambiano il risultato.

## 12. Test e criteri di correttezza

`runtime_metal_backend_test` copre:

- availability, lifecycle e nome del device;
- memoria, copie, pool e riuso;
- batch, sincronizzazione e metriche;
- elementwise, riduzioni e forme non multiple dei tile;
- FP32, FP16 e BF16 confrontati con il riferimento CPU;
- gather/scatter con indici duplicati e non validi;
- softmax e cross-entropy forward/backward;
- propagazione di input non finiti.

Il test salta l'esecuzione GPU quando Metal non e' disponibile, ma valida lo
stub. Su macOS la verifica autorevole va eseguita con accesso al device reale:

```sh
./build/release/tests/runtime_metal_backend_test
```

CPU e GPU possono differire negli ultimi bit per FMA, parallelismo e ordine
delle somme. I test usano tolleranze esplicite; non nascondono NaN o errori di
indice.

## 13. Limiti e prossimi incrementi

- kernel fusi dei futuri layer (normalizzazione, attention, MLP);
- cache persistente su disco delle scelte di autotuning;
- varianti SIMD-group anche per gli input FP16/BF16;
- memoria privata/heap per workload che non richiedono accesso host;
- profiling con Metal System Trace e counter set hardware;
- test su piu' generazioni Apple Silicon.

Questi limiti non impediscono di costruire i layer neurali, ma impediscono di
definire il runtime equivalente a un motore industriale completo. La base
attuale e' concreta, misurabile e abbastanza stabile da diventare il backend del
primo modello del progetto.
