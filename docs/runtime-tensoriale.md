# Runtime tensoriale — specifica implementativa

## 1. Obiettivo

Il runtime tensoriale e' lo strato numerico condiviso dai futuri layer del
language model. Deve rappresentare tensori, gestire la loro memoria ed eseguire
operazioni tramite un backend indipendente dall'hardware.

La prima implementazione usa FP32 per tutte le primitive e supporta anche
storage FP16/BF16, conversioni e matmul mista con accumulo FP32. Le API
permettono di scegliere CPU o GPU Metal senza modificare il codice del modello e
restano estendibili a CUDA.

Questa pagina definisce il contratto prima dell'implementazione. La spiegazione
introduttiva e' nella pagina wiki
[Runtime tensoriale: il motore di calcolo](../wiki/10-runtime-tensoriale.md).

**Stato:** gli Incrementi A-D sono implementati. Il backend CPU e' configurabile
e parallelo come specificato in [backend-cpu.md](backend-cpu.md). Il backend
Metal accelerato, i batch espliciti e le metriche sono descritti in
[backend-metal.md](backend-metal.md). Restano futuri autograd, layer neurali,
dispatch SIMD CPU avanzato, CUDA e fusion GPU.

## 2. Principi

Il runtime deve privilegiare, in questo ordine:

1. correttezza;
2. API piccole e leggibili;
3. ownership della memoria esplicita;
4. errori controllabili;
5. test indipendenti dal modello;
6. possibilita' di sostituire il backend;
7. ottimizzazione misurata, non anticipata.

Il codice di riferimento CPU deve restare comprensibile anche quando verranno
aggiunte implementazioni accelerate.

## 3. Confini

Il runtime v1 comprende:

- tipi comuni, dtype e dispositivi;
- storage e tensori;
- allocazione e copia della memoria;
- backend CPU e Metal, con batch asincroni espliciti su Metal;
- operazioni numeriche minime;
- kernel CPU di riferimento;
- validazione di forme, tipi e dispositivi;
- test di correttezza e gestione degli errori.

Non comprende ancora:

- layer neurali;
- grafo di autodifferenziazione;
- backward automatico;
- optimizer e training loop;
- CUDA o calcolo distribuito;
- FP8 e quantizzazione;
- scheduler asincrono generale e kernel fusi dei layer.

Questi elementi verranno costruiti sopra il contratto del runtime o aggiunti
come nuovi backend.

## 4. Architettura

```text
chiamante
   |
   v
API pubblica delle operazioni
   |  valida tensori, forme, dtype e dispositivi
   v
backend
   |  gestisce risorse e seleziona il kernel
   v
kernel del backend
   |  esegue il calcolo
   v
storage dei tensori di output
```

Il chiamante non deve includere header privati dei kernel o accedere a risorse
native del dispositivo.

## 5. Organizzazione proposta

Per mantenere il progetto facile da seguire, l'API pubblica e i contratti
privati sono divisi per responsabilita'. `runtime.h` resta un umbrella
compatibile per i chiamanti che desiderano l'intera API:

```text
include/runtime/
  types.h            status, dtype, device e tipi opachi
  backend.h          lifecycle e configurazione dei backend
  tensor.h           descrittore, ownership e memoria dei tensori
  operations.h       operazioni numeriche pubbliche
  runtime.h          umbrella dei quattro header pubblici

src/runtime/
  tensor.c           forma, stride e ciclo di vita
  memory.c           storage, allocazione e copie
  backend.c          creazione, capacita' e dispatch
  backend_internal.h contratto privato Runtime -> Backend
  storage_internal.h storage e allocazione privati
  tensor_internal.h  validazione tensoriale privata
  operations.c       controlli comuni delle operazioni
  backends/
    cpu/
      cpu_backend.c  risorse e dispatch CPU
      cpu_internal.h contratti privati CPU
      cpu_kernels.c  implementazioni numeriche CPU
      cpu_operations.c strategie parallele
      cpu_simd.c    SIMD baseline e fallback
      cpu_executor.c thread pool e parallel_for
      cpu_features.c rilevamento risorse CPU
      cpu_threads.c  portabilita' dei thread
    metal/
      metal_backend.m    dispositivo, pipeline e factory
      metal_memory.m     buffer e trasferimenti
      metal_operations.m dispatch delle operazioni
      metal_internal.h   contratti privati Metal
      kernels/
        runtime.metal    kernel Metal Shading Language

tests/runtime/
  test_runtime.c                 tensori, memoria e backend
  test_operations.c              elementwise, riduzioni e matmul
  test_language_operations.c     gather, softmax e cross-entropy
  test_metal_backend.c           disponibilita' e parita' CPU-Metal
```

Non verra' creato un file per ogni piccola operazione. Un file sara' diviso solo
quando dimensione e responsabilita' lo renderanno necessario. Ogni futuro
backend avra' una directory propria sotto `src/runtime/backends/`; non verranno
create directory vuote prima che esista una implementazione.

## 6. Tipi comuni

I nomi pubblici usano il prefisso `llm_` per evitare collisioni con librerie di
sistema.

### 6.1 Stato

Ogni operazione che puo' fallire restituisce uno stato:

```c
typedef enum llm_status {
    LLM_OK = 0,
    LLM_INVALID_ARGUMENT,
    LLM_INVALID_SHAPE,
    LLM_UNSUPPORTED_DTYPE,
    LLM_UNSUPPORTED_DEVICE,
    LLM_UNSUPPORTED_LAYOUT,
    LLM_DEVICE_MISMATCH,
    LLM_INVALID_INDEX,
    LLM_ALLOCATION_FAILED,
    LLM_OVERFLOW,
    LLM_NUMERICAL_ERROR,
    LLM_BACKEND_ERROR
} llm_status;
```

Una funzione `llm_status_string` restituisce una descrizione stabile. Il runtime
non stampa direttamente su standard output o standard error: decide il
chiamante come presentare l'errore.

### 6.2 Tipo numerico

```c
typedef enum llm_dtype {
    LLM_DTYPE_F32 = 0,
    LLM_DTYPE_U32,
    LLM_DTYPE_BF16,
    LLM_DTYPE_F16,
    LLM_DTYPE_F8_E4M3
} llm_dtype;
```

La presenza di un valore nell'enum non significa da sola che ogni operazione lo
supporti. FP32 e U32 coprono calcolo e indici; CPU e Metal supportano anche
storage F16/BF16, cast da/verso FP32 e matmul mista. FP8 resta riservato a un
incremento futuro.

### 6.3 Dispositivo

```c
typedef enum llm_device_type {
    LLM_DEVICE_NONE = 0,
    LLM_DEVICE_CPU,
    LLM_DEVICE_METAL,
    LLM_DEVICE_CUDA
} llm_device_type;
```

`LLM_DEVICE_NONE` descrive un backend o tensore vuoto. CPU e Metal sono
dispositivi operativi; CUDA riserva il valore per il futuro backend.

## 7. Storage e tensore

Storage e tensore sono concetti separati:

- lo **storage** possiede un buffer di memoria;
- il **tensore** descrive come interpretare una parte di quel buffer.

Questa separazione permettera' di creare viste senza duplicare i dati.

```c
#define LLM_TENSOR_MAX_RANK 4U

typedef struct llm_storage llm_storage;

typedef struct llm_tensor {
    llm_storage *storage;
    size_t offset;
    size_t rank;
    size_t shape[LLM_TENSOR_MAX_RANK];
    size_t strides[LLM_TENSOR_MAX_RANK];
    size_t element_count;
    llm_dtype dtype;
} llm_tensor;
```

Il dispositivo appartiene allo storage e si interroga tramite API; non viene
duplicato nel tensore.

Il gradiente non e' un campo del tensore. Quando introdurremo il backward, un
gradiente sara' un altro tensore con la stessa forma. Il runtime numerico resta
cosi' indipendente dall'autodifferenziazione.

### 7.1 Rank e forma

- rank `0`: scalare, con `element_count == 1`;
- rank `1`: vettore;
- rank `2`: matrice;
- rank `3`: per esempio `[batch, token, feature]`;
- rank `4`: per esempio `[batch, head, query, key]`.

Per rank maggiore di zero, ogni dimensione deve essere positiva. Il prodotto
delle dimensioni viene controllato prima di ogni moltiplicazione per impedire
overflow di `size_t`.

### 7.2 Layout

I tensori creati dalla v1 sono contigui e row-major. L'ultimo indice cambia piu'
velocemente:

```text
shape   = [2, 3]
strides = [3, 1]
```

`llm_tensor_reshape` crea una view contigua modificando shape e stride senza
duplicare i dati. `offset` resta zero. View non contigue, transpose e slice
verranno aggiunte soltanto quando richieste dal modello.

### 7.3 Ciclo di vita

API prevista:

```c
llm_status llm_tensor_create(llm_backend *backend, llm_dtype dtype,
                             size_t rank, const size_t *shape,
                             llm_tensor *out_tensor);

void llm_tensor_destroy(llm_tensor *tensor);

llm_status llm_tensor_move(llm_tensor *source,
                           llm_tensor *destination);

llm_status llm_tensor_reshape(const llm_tensor *input,
                              size_t rank,
                              const size_t *shape,
                              llm_tensor *out_view);

int llm_tensor_is_contiguous(const llm_tensor *tensor);

llm_device_type llm_tensor_device(const llm_tensor *tensor);
```

Il chiamante inizializza `out_tensor` a zero prima della creazione.
`llm_tensor_create` rifiuta un output che possiede gia' storage e produce un
tensore vuoto in caso di ogni altro errore.
`llm_tensor_destroy` accetta anche un tensore vuoto e lo azzera dopo il rilascio.
`llm_tensor_move` trasferisce l'ownership in una destinazione vuota e azzera la
sorgente, evitando copie proprietarie accidentali. `llm_tensor_reshape` richiede
un input contiguo, lo stesso numero di elementi e un output vuoto. Incrementa il
reference count dello storage; distruggere l'originale non invalida la view.

Una struttura `llm_tensor` proprietaria non deve essere copiata con una semplice
assegnazione C, perche' non incrementerebbe il reference count dello storage.
Il trasferimento usa `llm_tensor_move`; la condivisione usa una funzione di view
come `llm_tensor_reshape`.

## 8. Gestione della memoria

Il backend possiede l'allocatore. Lo storage registra almeno:

- backend proprietario;
- puntatore o handle nativo;
- dimensione in byte;
- allineamento;
- numero di riferimenti.

Per il backend CPU:

- le dimensioni vengono controllate prima dell'allocazione;
- il buffer e' allineato almeno a 64 byte quando la piattaforma lo permette;
- una richiesta di zero byte viene rifiutata;
- il rilascio e' sicuro su storage nullo;
- nessuna operazione perde l'ownership in caso di errore.

API prevista:

```c
llm_status llm_tensor_zero(llm_backend *backend, llm_tensor *tensor);
llm_status llm_tensor_fill_f32(llm_backend *backend, llm_tensor *tensor,
                               float value);
llm_status llm_tensor_copy(llm_backend *backend, const llm_tensor *source,
                           llm_tensor *destination);

llm_status llm_tensor_write(llm_backend *backend, llm_tensor *destination,
                            const void *source, size_t byte_count);

llm_status llm_tensor_read(llm_backend *backend, const llm_tensor *source,
                           void *destination, size_t byte_count);
```

La copia verifica forma, dtype e dimensione. I trasferimenti tra dispositivi
diversi verranno aggiunti insieme ai backend accelerati. `write` e `read`
trasferiscono l'intero payload tra memoria del chiamante e tensore; nella v1
`byte_count` deve coincidere esattamente con la dimensione del tensore.

## 9. Backend

Il backend pubblico e' opaco:

```c
typedef struct llm_backend llm_backend;
```

API minima:

```c
llm_status llm_backend_cpu_create(llm_backend **out_backend);
llm_status llm_backend_cpu_create_with_config(
    const llm_cpu_backend_config *config,
    llm_backend **out_backend);
size_t llm_backend_cpu_thread_count(const llm_backend *backend);
void llm_backend_destroy(llm_backend *backend);
llm_device_type llm_backend_device(const llm_backend *backend);
llm_status llm_backend_synchronize(llm_backend *backend);
```

La configurazione permette di scegliere il numero totale di thread; zero usa il
numero di processori rilevato entro il limite del backend. La CPU v1 e'
sincrona: quando una funzione restituisce `LLM_OK`, anche tutti i job paralleli
sono terminati. `llm_backend_synchronize` e' comunque presente nel contratto e
per la CPU restituisce immediatamente; i backend GPU potranno usarlo per
attendere una coda di lavoro asincrona.

La tabella di dispatch dei kernel resta privata. Il modello non puo' chiamare
direttamente `cpu_matmul` o funzioni native Metal/CUDA.

Il backend deve vivere piu' a lungo di tutti gli storage che ha allocato. Il
chiamante distrugge quindi prima i tensori e poi il backend. Ogni operazione
verifica che tutti i tensori appartengano proprio al backend ricevuto, non
soltanto a un dispositivo dello stesso tipo.

## 10. Operazioni della v1

Ogni operazione pubblica esegue prima gli stessi controlli:

1. puntatori validi;
2. storage esistente;
3. dtype supportato;
4. dispositivi compatibili;
5. forme compatibili;
6. layout supportato;
7. assenza di overflow nel calcolo degli offset.

Solo dopo il controllo viene chiamato il kernel.

Il chiamante crea anche il tensore di output con la forma prevista. Le operazioni
non allocano implicitamente il risultato: questa regola rende visibili costi,
ownership e durata della memoria.

La v1 non permette che l'output condivida lo storage con un input. Le future
operazioni esplicitamente in-place avranno nomi e contratti dedicati.

### 10.1 Operazioni di base

```c
llm_status llm_cast(llm_backend *backend, const llm_tensor *input,
                    llm_tensor *output);

llm_status llm_add(llm_backend *backend, const llm_tensor *left,
                   const llm_tensor *right, llm_tensor *output);

llm_status llm_multiply(llm_backend *backend, const llm_tensor *left,
                        const llm_tensor *right, llm_tensor *output);

llm_status llm_scale(llm_backend *backend, const llm_tensor *input,
                     float scale, llm_tensor *output);
```

`llm_cast` converte tra FP32 e FP16/BF16 a parita' di forma. Le operazioni
elementwise richiedono forme identiche e non implementano broadcasting
implicito. Evitare broadcasting inizialmente rende piu' chiari errori e regole
di memoria.

### 10.2 Moltiplicazione matriciale

```c
llm_status llm_matmul(llm_backend *backend, const llm_tensor *left,
                      const llm_tensor *right, llm_tensor *output);

llm_status llm_matmul_mixed_f32(llm_backend *backend,
                                const llm_tensor *left,
                                const llm_tensor *right,
                                llm_tensor *output);
```

Contratto v1:

```text
left   [M, K]
right  [K, N]
output [M, N]
```

`llm_matmul` richiede tensori FP32. `llm_matmul_mixed_f32` richiede due input
dello stesso tipo F16 o BF16 e un output FP32. Tutti i tensori sono contigui e
sullo stesso backend; ogni variante accumula in FP32. Ottimizzazioni BLAS o GPU
future resteranno dietro le stesse API.

### 10.3 Riduzioni

Le prime riduzioni operano sull'ultima dimensione:

```c
llm_status llm_reduce_sum_last(llm_backend *backend,
                               const llm_tensor *input,
                               llm_tensor *output);

llm_status llm_reduce_max_last(llm_backend *backend,
                               const llm_tensor *input,
                               llm_tensor *output);

llm_status llm_reduce_mean_square_last(llm_backend *backend,
                                       const llm_tensor *input,
                                       llm_tensor *output);
```

Specificare l'ultima dimensione evita nella v1 una API generica complessa. E' la
direzione necessaria per softmax e RMSNorm.

### 10.4 Embedding

```c
llm_status llm_gather_rows(llm_backend *backend,
                           const llm_tensor *table,
                           const llm_tensor *indices,
                           llm_tensor *output);

llm_status llm_scatter_add_rows(llm_backend *backend,
                                const llm_tensor *source,
                                const llm_tensor *indices,
                                llm_tensor *table);
```

`gather_rows` sara' usato dal forward dell'embedding. `scatter_add_rows` servira'
al backward e deve accumulare correttamente indici ripetuti.

`table` ha forma `[V, C]` e dtype FP32. `indices` usa U32; l'output aggiunge la
dimensione `C` alla forma degli indici. Ogni indice deve essere minore di `V`.
Tensori per gli indici permettono ai backend futuri di conservarli direttamente
sul dispositivo.

### 10.5 Softmax e cross-entropy

```c
llm_status llm_softmax_last(llm_backend *backend,
                            const llm_tensor *input,
                            llm_tensor *output);

llm_status llm_cross_entropy_forward(llm_backend *backend,
                                     const llm_tensor *logits,
                                     const llm_tensor *targets,
                                     llm_tensor *loss);

llm_status llm_cross_entropy_backward(llm_backend *backend,
                                      const llm_tensor *logits,
                                      const llm_tensor *targets,
                                      llm_tensor *logits_gradient);
```

La softmax sottrae il massimo di ogni riga prima di calcolare gli esponenziali.
La cross-entropy potra' usare un kernel fuso per non materializzare probabilita'
che non servono al chiamante.

Per la cross-entropy v1, `logits` ha forma `[N, V]` e dtype FP32, `targets` ha
forma `[N]` e dtype U32, `loss` e' uno scalare FP32 e `logits_gradient` ha la
stessa forma dei logits. Ogni target deve essere minore di `V`.

## 11. Kernel CPU di riferimento

I kernel CPU privati ricevono dati gia' validati. Non duplicano tutti i controlli
dell'API pubblica.

La prima versione deve essere:

- a singolo thread;
- deterministica con gli stessi input;
- priva di allocazioni nel ciclo interno;
- scritta con cicli espliciti;
- compilabile con Clang, GCC e MSVC;
- indipendente da librerie esterne.

Il kernel di riferimento non deve essere cancellato quando arrivera' una versione
ottimizzata: resta utile per test, confronto e studio.

## 12. Workspace

Alcune operazioni richiedono memoria temporanea. Non devono chiamare `malloc` per
ogni riga o per ogni passaggio del modello.

Le operazioni CPU v1 non richiedono workspace e non eseguono allocazioni durante
il calcolo. Un workspace esplicito verra' introdotto soltanto quando attenzione o
kernel accelerati avranno una necessita' misurata di memoria temporanea.

## 13. Relazione con backward e autograd

Il runtime esegue operazioni su tensori; non decide automaticamente quali
gradienti calcolare.

Il futuro sistema di backward registrera' operazioni come `matmul` e le visitera'
in ordine inverso. I gradienti saranno normali tensori:

```text
forward:  Y = X W
backward: dX = dY W^T
          dW = X^T dY
```

Questa separazione permette di testare prima i kernel e poi la composizione dei
gradienti.

## 14. Test richiesti

### Tensori

- creazione di scalare, vettore, matrice e tensore rank 4;
- calcolo corretto di `element_count` e stride;
- rifiuto di dimensioni zero e rank eccessivo;
- rilevamento degli overflow;
- distruzione ripetibile e sicura;
- distruzione corretta dello storage posseduto.

### Memoria

- buffer leggibile e scrivibile;
- zero, fill, read, write e copy corretti;
- rifiuto di copie con forma o dtype incompatibili;
- nessuna perdita nei percorsi di errore.

### Operazioni

- esempi piccoli con risultato noto;
- rifiuto di forme incompatibili;
- comportamento definito con `NaN` e `Inf`;
- softmax con righe che sommano circa a uno;
- softmax stabile con valori grandi;
- `scatter_add_rows` con indici duplicati;
- cross-entropy confrontata con un calcolo manuale.

### Backend

- creazione e distruzione;
- dispositivo dichiarato correttamente;
- sincronizzazione CPU valida;
- rifiuto di tensori appartenenti a un altro backend.
- rispetto della regola che il backend sopravvive ai suoi tensori.

In Debug i test devono essere eseguiti anche con AddressSanitizer quando la
toolchain lo permette. Le differenze in virgola mobile vengono confrontate con
tolleranze dichiarate dal singolo test, non con uguaglianza bit per bit.

## 15. Incrementi di implementazione

### Incremento A — Fondazioni

**Stato: implementato.**

- status, dtype e device;
- backend CPU;
- storage;
- creazione e distruzione dei tensori;
- forma, stride e overflow;
- test di memoria e ciclo di vita.

**Uscita:** e' possibile creare, riempire, copiare e distruggere un tensore FP32
su CPU senza perdite.

### Incremento B — Calcolo elementare

**Stato: implementato.**

- add, multiply e scale;
- riduzioni sull'ultima dimensione;
- test con risultati noti.

**Uscita:** le primitive elementwise e le riduzioni rispettano forme ed errori
documentati.

### Incremento C — Algebra lineare

**Stato: implementato nella versione CPU di riferimento.**

- matmul 2D di riferimento;
- casi rettangolari;
- controllo dell'aliasing;
- misurazione iniziale delle prestazioni.

**Uscita:** il backend calcola prodotti di matrici corretti e riproducibili.

### Incremento D — Primitive per il language model

**Stato: implementato.**

- gather e scatter-add;
- tensori U32 per gli ID;
- softmax stabile;
- cross-entropy forward e backward;
- assenza di allocazioni temporanee nelle primitive CPU v1.

**Uscita:** il runtime offre le operazioni minime per costruire embedding, layer
lineari e loss senza inserire cicli hardware-specifici nel modello.

### Incrementi futuri

- grafo e backward automatico;
- dispatch SIMD avanzato e BLAS CPU opzionale;
- scheduler Metal generale, fusion e autotuning persistente;
- backend CUDA;
- FP8 e quantizzazione;
- kernel fusi;
- esecuzione asincrona e distribuita.

## 16. Criterio di completamento della v1

Il runtime CPU v1 e' completo quando:

- tutte le API pubbliche hanno comportamento e ownership documentati;
- tutti i test richiesti passano su macOS, Linux e Windows;
- non esistono cicli numerici specifici della CPU nel futuro codice del modello;
- gli errori non lasciano oggetti parzialmente validi;
- il backend CPU rappresenta un riferimento confrontabile con backend futuri;
- un esempio minimo combina embedding, matmul e cross-entropy usando soltanto
  l'API pubblica.
