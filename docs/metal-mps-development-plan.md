# Piano di sviluppo — backend Metal ad alte prestazioni con MPS

**Stato (2026-08-13):** M0--M7 sono implementate nel codice e una sessione del
Modello Minimal da 2 milioni di step e' stata eseguita su Metal. M8--M10 richiedono ancora la
parita' contrattuale completa e benchmark riproducibili per forma.
**Destinatario:** sviluppatore del runtime Metal.
**Obiettivo:** completare il backend Metal v1 per il training, senza cambiare l'API pubblica C e senza delegare il modello a un framework esterno.

## 1. Decisione architetturale

Il progetto conserva il proprio runtime come livello di astrazione del modello:

```text
model / trainer
      |
      v
runtime API pubblica (llm_*)
      |
      +-- CPU: riferimento numerico e test
      |
      `-- Metal: adattatore a Metal + MPS + kernel custom
                       |
                       +-- MPS: GEMM e, dopo benchmark, softmax
                       `-- Metal kernel: operazioni Transformer specifiche
```

Il modello e il trainer non devono importare `Metal`, `MetalPerformanceShaders`, Objective-C o oggetti Apple. Devono conoscere soltanto `llm_backend`, `llm_tensor` e le operazioni di [`include/runtime/operations.h`](../include/runtime/operations.h).

Il backend Metal e' quindi un **adattatore interno**: per ogni operazione decide se usare un kernel Metal locale o una primitiva Apple, ma produce esattamente la semantica del contratto [Runtime v1](runtime-v1-architecture.md).

### Stato di implementazione

Le milestone M0--M7 sono implementate, inclusi RMSNorm, RoPE, attention GQA e
AdamW. La build Release e i test Metal dedicati sono stati eseguiti su un Mac
mini Apple M4 con 24 GiB di memoria unificata; il GEMM F32 `512x512x512` ha
misurato circa 360 GFLOP/s Metal end-to-end contro circa 95 GFLOP/s CPU a 10
thread. Una successiva sessione del Modello Minimal su Metal ha raggiunto 2
milioni di step.

Restano M8 (parita' contrattuale completa), M9 (profiling e tuning per forma) e
la dimostrazione formale M10. CUDA resta deliberatamente fuori dallo scope
finche' non sono disponibili hardware e CI per validarlo.

### Decisioni vincolanti

- Mantenere F32 per i valori e U32 per gli indici: niente F16/BF16, cast o mixed precision in questa fase.
- Non introdurre fallback CPU: un'operazione richiesta a Metal resta sul device oppure restituisce `LLM_UNSUPPORTED_OPERATION` / `LLM_BACKEND_ERROR`.
- Mantenere gli attuali buffer `MTLBuffer`, command queue, batch esplicito e buffer pool.
- Usare MPS come dettaglio implementativo, non come dipendenza dell'API pubblica.
- Non usare MPSGraph o MLX nel backend v1. Entrambi sono validi framework di alto livello, ma si sovrappongono al runtime, al backward esplicito e al trainer che il progetto vuole possedere e testare.
- Introdurre Metal Performance Primitives / TensorOps soltanto come percorso sperimentale opzionale e capability-gated: richiede un target OS/GPU piu' recente e non deve diventare obbligatorio per gli Apple Silicon gia' supportati.

## 2. Cosa usare di Apple e cosa mantenere custom

| Famiglia | Prima implementazione | Percorso prestazionale | Motivazione |
|---|---|---|---|
| `matmul`, `matmul_ex` | `MPSMatrixMultiplication` | MPS oppure kernel `simdgroup` gia' presente, scelti per shape | MPS gestisce GEMM F32 e entrambe le trasposizioni; il kernel custom puo' vincere su forme piccole o molto regolari. |
| `softmax` | kernel Metal attuale | confrontare con `MPSMatrixSoftMax` | Il kernel locale puo' evitare wrapper MPS e adattarsi alla semantica del runtime; MPS va adottato solo se vince nei benchmark. |
| add/multiply/scale/riduzioni | kernel Metal attuali | fusioni future | Dispatch MPS separati per operazioni minuscole possono essere piu' lenti. |
| gather/scatter embedding | kernel Metal custom | kernel custom migliorato | Gli indici U32 e la scatter-add con collisioni sono parte della semantica del progetto. |
| SiLU, RMSNorm, RoPE | kernel Metal custom | fusioni mirate | Sono operazioni Transformer con backward e layout esatti propri. |
| attention causale GQA | kernel Metal custom | kernel tiled/fused | La causalita', il mapping GQA e il backward non coincidono con una chiamata MPS generica. |
| cross entropy e AdamW | kernel Metal custom | kernel custom/fuso | Necessitano precisione, riduzioni e aggiornamenti in-place conformi al contratto. |

`MPSMatrix` e' row-major e puo' essere creato sopra un `MTLBuffer` esistente. Questo permette di usare direttamente i buffer del runtime senza copie. Attenzione: `rowBytes` MPS puo' avere padding; il runtime espone tensori contigui senza padding. Per la prima integrazione impostare `rowBytes = columns * sizeof(float)` e abilitare MPS solo quando tale valore e' accettato e il buffer ha capienza corretta. Non riallocare o repackare un tensore solo per soddisfare MPS.

Riferimenti: [MPS](https://developer.apple.com/documentation/MetalPerformanceShaders), [MPSMatrix](https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrix), [MPSMatrixMultiplication](https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrixmultiplication), [MPSMatrixSoftMax](https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrixsoftmax).

## 3. File coinvolti e responsabilita'

| File | Responsabilita' dopo il lavoro |
|---|---|
| `CMakeLists.txt` | trovare e collegare `MetalPerformanceShaders.framework` solo su Apple. |
| `src/runtime/backends/metal/metal_internal.h` | pipeline, cache MPS, capability e prototipi Metal privati. |
| `src/runtime/backends/metal/metal_backend.m` | creazione/distruzione del contesto, vtable completa, scelta device e cache. |
| `src/runtime/backends/metal/metal_operations.m` | wrapper ObjC: validazione interna, encoding MPS/Metal, dispatch e scelta del percorso. |
| `src/runtime/backends/metal/kernels/runtime.metal` | kernel e struct parametri custom. |
| `src/runtime/backends/metal/kernels/matmul_simdgroup.metal` | GEMM custom, mantenuto come candidato alternativo a MPS. |
| `tests/runtime/backend_contract_suite.c` | sorgente condivisa della correttezza CPU/Metal; estendere solo per nuove casistiche contrattuali. |
| `tests/runtime/test_metal_backend.c` | disponibilita', batch, metriche, errori e test specifici MPS. |
| `utils/benchmarks/runtime/*` | benchmark comparabili per scegliere MPS o kernel custom. |

Non cambiare `include/runtime/*.h` per esporre oggetti MPS, soglie di tuning o interruttori Apple. Se occorre un knob per sviluppo, usare una variabile d'ambiente privata e non documentarla come API stabile; rimuoverla o renderla diagnostica prima del merge.

## 4. Preparazione della build

### 4.1 Collegare il framework

In `CMakeLists.txt`, nel ramo `if(APPLE)`, aggiungere:

```cmake
find_library(METAL_PERFORMANCE_SHADERS_FRAMEWORK MetalPerformanceShaders REQUIRED)
target_link_libraries(runtime PUBLIC ${METAL_PERFORMANCE_SHADERS_FRAMEWORK})
```

`metal_backend.m` e `metal_operations.m` sono gia' Objective-C; importare nel file che usa MPS:

```objective-c
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
```

Non includere MPS negli header pubblici. Il backend stub Linux resta invariato e continua a restituire `LLM_UNSUPPORTED_DEVICE`.

### 4.2 Verifiche iniziali

1. Configurare Debug su un Mac Apple Silicon: `cmake --preset debug`.
2. Compilare: `cmake --build --preset debug`.
3. Eseguire baseline: `ctest --test-dir build/debug --output-on-failure`.
4. Eseguire il benchmark Metal prima di cambiare codice e salvare il JSONL come baseline locale.
5. Annotare: modello Mac, versione macOS, Xcode/Metal compiler, device name e memoria unificata. I risultati di tuning non sono trasferibili automaticamente a tutte le GPU Apple.

## 5. Estensioni interne proposte

### 5.1 Vtable: completarla, non modificarla

La struttura `llm_backend_ops` gia' dichiara tutti gli slot necessari. In `metal_backend.m`, completare la vtable con questi puntatori:

```objective-c
.matmul_ex_f32 = llm_metal_matmul_ex_f32,
.accumulate_f32 = llm_metal_accumulate_f32,
.silu_f32 = llm_metal_silu_f32,
.silu_backward_f32 = llm_metal_silu_backward_f32,
.rms_norm_f32 = llm_metal_rms_norm_f32,
.rms_norm_backward_f32 = llm_metal_rms_norm_backward_f32,
.rope_f32 = llm_metal_rope_f32,
.rope_backward_f32 = llm_metal_rope_backward_f32,
.attention_forward_f32 = llm_metal_attention_forward_f32,
.attention_backward_f32 = llm_metal_attention_backward_f32,
.adamw_update_f32 = llm_metal_adamw_update_f32,
```

Le firme esatte devono coincidere con `src/runtime/backend_internal.h`. Non aggiungere autodiff, tensor view o un dispatcher nell'implementazione Metal.

### 5.2 Pipeline da aggiungere

Estendere l'enum `llm_metal_pipeline` in `metal_internal.h` e mantenere lo stesso ordine nella tabella di nomi/pipeline creata in `metal_backend.m`:

```text
ACCUMULATE
MATMUL_EX_FALLBACK       (facoltativa: una sola pipeline parametrica e' preferibile)
SILU
SILU_BACKWARD
RMS_NORM
RMS_NORM_BACKWARD_INPUT
RMS_NORM_BACKWARD_WEIGHT
ROPE
ROPE_BACKWARD
ATTENTION_FORWARD
ATTENTION_BACKWARD_DV
ATTENTION_BACKWARD_DQ_DK
ADAMW
```

Dividere RMSNorm backward e attention backward in piu' kernel e' corretto: i gradienti sono output distinti e richiedono riduzioni/accumuli diversi. Le chiamate pubbliche restano una sola; il wrapper encoda piu' dispatch nello stesso command buffer.

### 5.3 Cache MPS privata

Non creare un `MPSMatrixMultiplication` per ogni chiamata. Aggiungere strutture private, per esempio:

```c
typedef struct llm_metal_mps_gemm_key {
    uint32_t left_rows, left_columns;
    uint32_t right_rows, right_columns;
    uint8_t transpose_left, transpose_right;
} llm_metal_mps_gemm_key;

typedef struct llm_metal_mps_gemm_entry {
    llm_metal_mps_gemm_key key;
    id<MPSMatrixMultiplication> kernel;
    struct llm_metal_mps_gemm_entry *next;
} llm_metal_mps_gemm_entry;
```

Nel contesto aggiungere una lista/cache con limite (per esempio 128 chiavi), protetta dallo stesso mutex o da un mutex dedicato. Distruggere ogni oggetto in `metal_destroy`. Non mettere `MPSMatrix` nella cache: il wrapper contiene uno specifico `MTLBuffer` e va creato per la chiamata; la pipeline MPS e' invece riusabile per shape e trasposizioni uguali.

Funzioni private consigliate:

```objective-c
static MPSMatrix *metal_wrap_matrix(id<MTLBuffer> buffer, size_t offset,
                                    size_t rows, size_t columns);
static id<MPSMatrixMultiplication> metal_mps_gemm_get_or_create(
    llm_metal_context *context, size_t left_rows, size_t left_columns,
    size_t right_rows, size_t right_columns, int transpose_left, int transpose_right);
static llm_status metal_mps_encode_matmul_ex(...);
static llm_metal_pipeline metal_choose_matmul_path(...);
```

`metal_wrap_matrix` deve controllare overflow da `size_t` a `NSUInteger`, contiguita' (`offset == 0` nell'attuale storage), `MPSDataTypeFloat32`, `rows > 0`, `columns > 0` e capienza del buffer. Deve vivere dentro un `@autoreleasepool` che duri fino a dopo `encode...`.

## 6. Implementazione per fasi

Ogni fase deve terminare con build, CTest e benchmark; non iniziare l'ottimizzazione della fase successiva se il confronto CPU/Metal non e' verde.

### Fase A — GEMM MPS e `matmul_ex`

**Obiettivo:** supportare tutte le quattro combinazioni di trasposizione con `C = op(A) * op(B)`.

1. Aggiungere il framework MPS alla build e verificare l'avvio del backend.
2. Implementare `llm_metal_matmul_ex_f32` in `metal_operations.m` con la firma della vtable.
3. Calcolare le dimensioni logiche:

   ```text
   M = transpose_left  ? left_columns : left_rows
   K = transpose_left  ? left_rows    : left_columns
   K = transpose_right ? right_columns: right_rows
   N = transpose_right ? right_rows   : right_columns
   ```

   La facciata comune valida la compatibilita'; il backend ricontrolla overflow di conversione e capienza.
4. Creare tre `MPSMatrix` sugli stessi `MTLBuffer` dei tensori. `alpha = 1`, `beta = 0`; `beta` diverso da zero non appartiene alla semantica pubblica.
5. Ottenere dalla cache `MPSMatrixMultiplication` configurato con `transposeLeft`, `transposeRight`, `M`, `N`, `K`.
6. Codificare il kernel MPS nel command buffer restituito da `llm_metal_acquire_command_buffer`, quindi chiamare `llm_metal_submit`. Non creare un `MTLComputeCommandEncoder` per MPS: MPS encoda direttamente nel command buffer; se era aperto un encoder batch, chiuderlo prima di MPS e lasciar riprendere l'encoder per il successivo kernel custom.
7. Dentro batch esplicito, l'operazione non deve fare `commit`/`wait`: deve essere accodata nello stesso command buffer. Eventuali passaggi fra encoder Metal e MPS richiedono solo la corretta chiusura dell'encoder, non una sincronizzazione.
8. Conservare inizialmente il kernel custom gia' esistente come alternativa. La selezione puo' partire semplice: MPS per `M*N*K >= soglia`, custom sotto soglia. La soglia viene decisa solo dopo benchmark.

**Test obbligatori:** forme 1x1, 1xN, Mx1, K non multiplo di 16/32, tutte le trasposizioni, output pre-riempito (deve venire sovrascritto), batch attivo, due GEMM consecutivi e confronto CPU. Aggiungere una regressione che controlli che il numero di command buffer in un batch non cresca per ogni GEMM.

### Fase B — primitive semplici e SiLU

1. Implementare `llm_metal_accumulate_f32`: kernel elementwise `destination[i] += source[i]` con buffer distinti.
2. Aggiungere in `runtime.metal`:

   ```metal
   kernel void llm_silu_f32(...);
   kernel void llm_silu_backward_f32(...);
   ```

3. Forward: `y = x / (1 + exp(-x))`. Backward: dato `g`, calcolare `g * sigmoid(x) * (1 + x * (1 - sigmoid(x)))` in F32.
4. Per valori grandi usare una forma numericamente stabile di sigmoid, coerente con la CPU entro tolleranza.
5. Per ora non usare MPS per SiLU: un kernel proprio evita conversioni e permette in futuro fusioni.

**Test obbligatori:** valori negativi/grandi, dimensioni non multiple della vector width, gradiente numerico e confronto CPU/Metal.

### Fase C — RMSNorm forward/backward

**Forward per ogni riga** `x[0..C-1]`:

```text
mean_square = sum(x[i]^2) / C
inv_rms     = 1 / sqrt(mean_square + epsilon)
y[i]       = x[i] * inv_rms * weight[i]
```

Implementazione consigliata:

1. Un threadgroup per riga `[outer_count, C]`; riduzione threadgroup per `sum(x^2)`.
2. Ogni thread calcola piu' elementi stridati; usare `simd_sum` e memoria threadgroup, come nelle riduzioni esistenti.
3. Una volta ottenuto `inv_rms`, ogni thread scrive la sua parte dell'output.
4. Backward input: un threadgroup per riga. Calcolare le somme necessarie, poi `dx` per elemento.
5. Backward weight: non aggiornare una tabella globale con molte `atomic` per ogni elemento se si puo' evitare. Prima versione corretta: kernel per blocchi di righe che produce gradienti parziali `[chunk, C]`, poi un kernel di riduzione che scrive `weight_gradient[C]`. Questo e' deterministico per ordine di riduzione fissato e scala meglio delle contese atomiche.
6. Allocare il buffer temporaneo tramite il buffer pool. Deve restare sul device e rientrare nel pool solo dopo il command buffer.

**Test obbligatori:** `outer_count` non multiplo, `C = 2`, `C` dispari, epsilon piccolo positivo, confronto di `output`, `input_gradient`, `weight_gradient` e differenze finite.

### Fase D — RoPE forward/backward

Un thread gestisce una coppia di canali adiacenti. Per `[B,S,H,D]`, il numero logico di coppie e' `B*S*H*(D/2)`.

```text
forward:  y0 = x0*cos - x1*sin;  y1 = x0*sin + x1*cos
backward: dx0 = g0*cos + g1*sin; dx1 = -g0*sin + g1*cos
```

1. Aggiungere struct parametri con `B,S,H,D` come U32 dopo aver verificato overflow.
2. Derivare dal linear index: coppia, head, posizione e batch; leggere `cos[position, dimension/2]` e `sin[...]`.
3. Non supportare offset, cache o valori D dispari: sono gia' rifiutati dalla facciata.
4. Usare un solo dispatch per forward e uno per backward.

**Test obbligatori:** `S` e `D` non multipli del tile, `D=2`, molte head, segno del backward, confronto CPU.

### Fase E — attention causale GQA forward

La prima versione privilegia correttezza e memoria O(1) aggiuntiva rispetto a Q/K/V/output: non materializzare l'intera matrice score `[B,Hq,S,S]`.

Per ogni output `(b, q_position, q_head)`:

1. Mappare `kv_head = q_head / (Hq / Hkv)`.
2. Una query vede solo `key_position <= q_position`.
3. Calcolare online max, somma e numeratore dell'output con l'algoritmo softmax stabile (online softmax):

   ```text
   m = -inf; l = 0; acc[D] = 0
   per k <= q:
       score = scale * dot(Q[q], K[k])
       new_m = max(m, score)
       a = exp(m - new_m); b = exp(score - new_m)
       acc = acc*a + V[k]*b
       l = l*a + b; m = new_m
   out = acc / l
   ```

4. Per la prima implementazione usare un threadgroup per `(b,q_position,q_head)` e parallelizzare il dot su `D`; se `S` e' grande, fare tile su K. Tenere `acc[D]` in registri quando `D` e' piccolo o in threadgroup memory quando necessario.
5. Validare prestazioni su `D=32,64,128`, `S=32..512`, GQA e MHA. Non assumere che un solo tile sia ottimale.

### Fase F — attention backward

Il backward puo' ricalcolare score/probabilita', come autorizzato dal contratto. Non copiare score o probabilita' su host.

Proposta in due passaggi per query/head:

1. Calcolare `P` online, `dP[k] = dot(dO, V[k])` e `delta = sum(P[k] * dP[k])`.
2. Per ogni key causale calcolare `dS[k] = P[k] * (dP[k] - delta)`, quindi:

   ```text
   dQ += scale * dS[k] * K[k]
   dK += scale * dS[k] * Q
   dV += P[k] * dO
   ```

3. `dQ` ha un solo produttore per `(b,q,h)`; scriverlo direttamente. `dK` e `dV` ricevono contributi da molte query e da head GQA: prima versione corretta usa atomic float add o buffer parziali. Per le prestazioni, preferire buffer parziali per tile di query e riduzione finale deterministica.
4. Azzerare gradienti e temporanei sul device; non fare `read` per controllare valori intermedi.

**Test obbligatori:** causalita' (alterare token futuro non cambia output precedente), `Hq % Hkv == 0`, MHA (`Hq==Hkv`) e GQA, `S=1`, dimensioni non multiple, finite-difference su tensori piccoli e confronto CPU con tolleranza dichiarata.

### Fase G — AdamW

Un kernel elementwise, senza atomiche:

```text
g = gradient * gradient_scale
m = beta1*m + (1-beta1)*g
v = beta2*v + (1-beta2)*g*g
p = p*(1-learning_rate*weight_decay)
    - learning_rate*(m/(1-beta1^step))/(sqrt(v/(1-beta2^step))+epsilon)
```

1. Passare iperparametri in una struct Metal allineata a 16 byte; `step` e' `uint64_t`/`ulong` sia in Objective-C sia in Metal.
2. La facciata ha gia' validato iperparametri e alias; il backend verifica che i buffer appartengano al contesto.
3. Poiche' l'operazione deve segnalare `LLM_NUMERICAL_ERROR` se non puo' produrre output finiti, evitare una scansione CPU dei buffer. Prima soluzione GPU-only: un buffer flag U32 azzerato, ogni thread fa `isfinite` su valori finali e imposta atomicamente il flag; un singolo read del flag dopo completamento e' ammesso solo fuori batch. In batch, registrare il flag e restituire l'errore a `end_batch`; documentare e testare questa semantica.
4. Se si preferisce mantenere la semantica sincrona anche in batch, disabilitare temporaneamente batch per AdamW non e' accettabile: spezzerebbe il training step. Implementare la propagazione differita del flag.

## 7. Gestione corretta dei command buffer

Regola centrale: una sequenza interna Metal/MPS/Metal e' una sequenza di encoder nello **stesso** `MTLCommandBuffer`.

```text
Metal compute encoder -> endEncoding
MPS encode(commandBuffer, ...) -> nessun wait
Metal compute encoder -> endEncoding
commit solo alla fine della chiamata sincrona o in end_batch
```

- Fuori batch, `llm_metal_submit` puo' commit+wait per mantenere l'osservabilita' sincrona indicata dalla specifica.
- Dentro batch, `llm_metal_submit` non deve committare; `llm_backend_metal_end_batch` chiude l'encoder, committa e attende.
- Non leggere `-[MTLBuffer contents]` per calcolare risultati o implementare operazioni; i soli casi ammissibili sono write/read pubblici, diagnostica esplicita e il flag numerico minimalmente necessario.
- Ogni buffer temporaneo deve essere trattenuto fino al completamento del command buffer. Se il pool ricicla un buffer troppo presto, introdurre una lista di buffer "in flight" rilasciata nel completion handler.

## 8. Selezione del percorso piu' veloce

Non decidere che MPS e' sempre piu' veloce. Implementare un selettore solo per GEMM:

```text
chiave = (M, K, N, transpose_left, transpose_right, device family)
scelta = MPS | SIMDGROUP | kernel tiled base
```

Percorso iniziale semplice e riproducibile:

1. `matmul_ex` con trasposizioni usa MPS.
2. `matmul` senza trasposizioni conserva l'attuale autotuning fra kernel locali.
3. Aggiungere benchmark per una griglia rappresentativa delle dimensioni del futuro modello.
4. Promuovere MPS al percorso standard soltanto per le shape in cui supera il miglior kernel custom di almeno una soglia misurata (per esempio 5%, decisa e documentata insieme ai dati).
5. Registrare la scelta per chiave nel contesto; non fare benchmark durante una sessione reale di training salvo modalita' esplicitamente "autotune".

Forme minime da misurare: `M=B*S` in `{16,64,256,1024}`, `K,N` in `{128,256,512,1024,2048,4096}`, trasposizioni tipiche del backward e dimensioni non multiple di 16/32. Misurare anche le forme concrete del primo modello, non solo quadrate.

## 9. Test e criteri numerici

### 9.1 Strategia

La CPU e' il riferimento. I test devono scrivere gli stessi input su CPU e Metal, sincronizzare Metal soltanto al punto di confronto e verificare output/gradienti con tolleranza assoluta e relativa per operazione.

Suggerimento iniziale, da confermare con risultati reali:

| Operazione | Tolleranza iniziale |
|---|---:|
| elementwise, RoPE, AdamW | `atol=1e-5`, `rtol=1e-5` |
| GEMM, SiLU, RMSNorm | `atol=2e-4`, `rtol=2e-4` |
| attention e relativi gradienti | `atol=1e-3`, `rtol=1e-3` |

Non allentare la tolleranza per far passare un errore sistematico; prima localizzare se e' mapping, causalita', riduzione o precisione.

### 9.2 Gate di ogni pull request

- Build Debug senza warning.
- `ctest --test-dir build/debug --output-on-failure` verde su macOS Apple Silicon.
- Test Metal saltati correttamente quando non c'e' device e realmente eseguiti su un Mac con GPU Metal.
- Nuovi test per forme limite, forme irregolari, batch esplicito e confronto CPU/Metal.
- Nessun cambiamento di API pubblica non richiesto.
- Nessun host round-trip nel percorso misurato; verificarlo con metriche dispatch/command buffer e Metal System Trace quando appropriato.
- Benchmark prima/dopo allegato quando viene cambiato il percorso di una operazione calda.

## 10. Piano di integrazione e milestone

| Milestone | Deliverable | Gate |
|---|---|---|
| M0 | framework MPS collegato, baseline e metriche | build/test invariati |
| M1 | `matmul_ex` con MPS, batch corretto | suite GEMM CPU/Metal |
| M2 | accumulate e SiLU forward/backward | test e gradient check |
| M3 | RMSNorm forward/backward | test gradienti e no temporanei host |
| M4 | RoPE forward/backward | test semantici e forme irregolari |
| M5 | attention GQA forward | causalita' e confronto CPU |
| M6 | attention GQA backward | gradient check e suite contrattuale |
| M7 | AdamW e segnalazione numerica | test valori/flag/batch |
| M8 | parita' runtime v1 completa | `backend_contract_suite` Metal verde |
| M9 | tuning MPS/custom e profiling | benchmark riproducibili, nessuna regressione di correttezza |
| M10 | smoke model CPU/Metal | un training step interamente on-device |

La milestone M8 conclude il backend. Solo allora il modello e il trainer devono dipendere da Metal come backend di training ufficiale. M10 non richiede un trainer completo: basta embedding, una proiezione lineare, cross-entropy, backward e AdamW per dimostrare che un passo evita ogni trasferimento intermedio host.

## 11. Cose da non fare

- Non cambiare silenziosamente il contratto a F16 per ottenere numeri di benchmark migliori.
- Non richiamare la CPU dal codice Objective-C per transposte, riduzioni, gradienti o controlli intermedi.
- Non fare una copia in un buffer MPS separato a ogni GEMM: annullerebbe gran parte del vantaggio della memoria unificata.
- Non materializzare score/probabilita' dell'attention su `[B,H,S,S]` nella prima implementazione, salvo un test minuscolo di debug: la memoria cresce troppo rapidamente.
- Non introdurre KV cache, decode incrementale, quantizzazione, CUDA o MPSGraph durante questa milestone.
- Non fondere molte operazioni prima di avere una versione non fusa, testata contro CPU: la fusion e' un passaggio di ottimizzazione, non di specifica.

## 12. Definition of Done

Il backend Metal e' pronto per costruire il modello quando tutti i seguenti punti sono veri:

1. ogni slot vtable previsto da `runtime-v1-architecture.md` ha un'implementazione Metal;
2. la suite contrattuale condivisa passa su hardware Apple reale;
3. il backend usa MPS per le GEMM per cui i benchmark lo indicano migliore, e kernel propri per il resto;
4. un batch di training esegue forward, backward e AdamW senza `llm_tensor_read/write` intermedi;
5. output e gradienti del piccolo smoke model coincidono con CPU nelle tolleranze dichiarate;
6. metriche e benchmark documentano i percorsi scelti e le performance su almeno una macchina target;
7. build Linux continua a funzionare grazie allo stub Metal e nessuna API Apple e' filtrata fuori dal backend.

Da quel punto si passa a `model` e `trainer`: il loro codice resta portabile e non deve essere modificato se, in futuro, un kernel MPS/Metal viene sostituito da una fusione piu' veloce.
