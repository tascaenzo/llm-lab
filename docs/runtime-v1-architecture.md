# Runtime v1 — minimo necessario per addestrare su Apple Silicon

**Stato:** contratto Runtime v1; milestone CPU F32 completata.
**Target hardware:** Apple Silicon, con CPU come riferimento e Metal come backend di training.
**Verifica del codice:** 11 agosto 2026.

## 1. Obiettivo verificabile

Il Runtime v1 e' completo quando un piccolo decoder-only Transformer puo':

1. eseguire il forward;
2. calcolare la cross-entropy;
3. eseguire il backward di tutti i parametri;
4. aggiornare i pesi con AdamW;
5. ridurre realmente la loss su un piccolo dataset;
6. compiere l'intero training step su Metal senza fallback CPU nascosti.

Il modello target usa soltanto:

```text
token embedding
RMSNorm
RoPE
Grouped-Query causal attention
SwiGLU
residual connection
vocabulary projection
cross-entropy
AdamW
```

CPU e Metal devono offrire la stessa semantica. CPU e' l'oracolo numerico e il
percorso di debug; Metal e' il percorso principale di addestramento sulla GPU
Apple.

## 2. Glossario semplice

Questa sezione spiega i termini usati nel documento. Non e' necessario
conoscerne subito i dettagli matematici: per iniziare basta comprenderne il
ruolo nel percorso dei dati.

### 2.1 Componenti del modello

| Termine | Descrizione semplice |
|---|---|
| **LLM (Large Language Model)** | Modello che impara a prevedere il token successivo osservando i token precedenti. |
| **Token** | Piccola unita' di testo. Puo' essere una parola, una parte di parola, un simbolo o uno spazio. |
| **Transformer decoder-only** | Rete composta da blocchi ripetuti che leggono solo il testo passato e prevedono il seguito. “Decoder-only” e' l'architettura usata da molti modelli generativi. |
| **Embedding** | Tabella che trasforma l'identificatore numerico di un token in un vettore di numeri appreso durante il training. |
| **Hidden state** | Rappresentazione numerica corrente di ogni token mentre attraversa i layer del modello. |
| **Layer o decoder block** | Un blocco ripetuto del Transformer. Contiene normalmente attention, normalizzazione, MLP e connessioni residuali. |
| **RMSNorm** | Normalizza la grandezza di ogni hidden state e poi applica un peso appreso. Mantiene i valori numericamente stabili tra i layer. |
| **RoPE (Rotary Position Embedding)** | Inserisce l'informazione sulla posizione ruotando coppie di valori nelle query e nelle key. Permette all'attention di distinguere l'ordine dei token. |
| **Attention** | Per ogni token calcola quanto sono importanti i token precedenti e ne combina le informazioni. |
| **Attention causale** | Variante che impedisce a un token di vedere i token futuri. E' necessaria per imparare a prevedere il token successivo senza “barare”. |
| **Query, Key e Value (Q, K, V)** | Tre rappresentazioni ricavate dagli hidden state. Query e Key determinano quanto due token sono collegati; Value contiene l'informazione da trasferire. |
| **Attention head** | Una prospettiva indipendente con cui il modello cerca relazioni fra token. Piu' head possono imparare tipi di relazione differenti. |
| **GQA (Grouped-Query Attention)** | Piu' query head condividono un numero minore di key/value head. Riduce memoria e calcolo mantenendo molte query head. |
| **Softmax** | Trasforma una riga di punteggi in pesi positivi che sommano a uno. Nell'attention indica quanta importanza assegnare a ciascun token. |
| **Scaling dell'attention** | Riduce i punteggi prima della softmax, normalmente dividendoli per la radice della dimensione della head. Evita valori troppo grandi e training instabile. |
| **MLP (Multi-Layer Perceptron)** | Parte del blocco che elabora separatamente il vettore di ogni token tramite proiezioni lineari e un'attivazione. |
| **Sigmoid** | Funzione che trasforma qualsiasi numero in un valore fra zero e uno. In SiLU viene usata come un piccolo regolatore del segnale. |
| **SiLU** | Funzione di attivazione morbida: lascia passare o attenua ogni valore. In formula semplice, `SiLU(x) = x * sigmoid(x)`. |
| **SwiGLU** | MLP con due rami: un ramo crea un gate tramite SiLU e controlla quanto dell'altro ramo deve passare. |
| **Connessione residuale** | Somma l'input di un blocco al suo output. Aiuta informazione e gradienti ad attraversare molti layer. |
| **Proiezione lineare** | Moltiplicazione per una matrice di pesi appresa. Cambia la dimensione o il significato di una rappresentazione. |
| **Proiezione sul vocabolario** | Ultima proiezione: trasforma l'hidden state in un punteggio per ogni token possibile del vocabolario. |
| **Logit** | Punteggio grezzo assegnato dal modello a un possibile token, prima della softmax. Un logit maggiore indica una preferenza maggiore. |

### 2.2 Addestramento

| Termine | Descrizione semplice |
|---|---|
| **Forward** | Passaggio dagli input alle predizioni. Calcola hidden state, logits e loss usando i pesi correnti. |
| **Loss** | Singolo numero che misura quanto sono sbagliate le predizioni. Il training cerca di ridurlo. |
| **Cross-entropy** | Loss che confronta i logits con il token corretto. Penalizza il modello quando assegna poca probabilita' alla risposta giusta. |
| **Backward o backpropagation** | Percorre i calcoli al contrario per determinare come ogni peso ha contribuito all'errore. |
| **Gradiente** | Numero, o tensore di numeri, che indica in quale direzione modificare un peso per ridurre la loss. |
| **Gradient accumulation** | Somma gradienti ottenuti da piu' micro-batch prima di aggiornare i pesi. Permette di simulare batch grandi usando meno memoria. |
| **Gradient clipping** | Limita gradienti eccessivamente grandi per evitare aggiornamenti instabili. Non e' richiesto nel primo milestone, ma verra' aggiunto dopo il training F32. |
| **AdamW** | Algoritmo che aggiorna i pesi usando medie mobili dei gradienti e applica weight decay. Adatta automaticamente la dimensione degli aggiornamenti. |
| **Weight decay** | Piccola spinta che evita che i pesi crescano troppo. AdamW la applica separatamente dal gradiente. |
| **Learning rate** | Intensita' con cui AdamW modifica i pesi a ogni step. Troppo alto rende il training instabile; troppo basso lo rende lento. |
| **Training step** | Un ciclo completo: forward, loss, backward e aggiornamento AdamW. |
| **Batch** | Gruppo di esempi elaborati insieme in un training step. |
| **Micro-batch** | Parte piu' piccola di un batch. I suoi gradienti vengono accumulati prima dell'aggiornamento. |
| **Gradient check numerico** | Test che modifica leggermente un valore e controlla che il gradiente calcolato dal backward sia corretto. E' lento ma molto utile per trovare errori. |
| **Overfittare un batch fisso** | Test iniziale in cui il modello vede ripetutamente gli stessi pochi esempi. Se non riesce ad abbassare molto la loss, probabilmente training o backward contengono un errore. |

### 2.3 Calcolo tensoriale e hardware

| Termine | Descrizione semplice |
|---|---|
| **Tensore** | Contenitore multidimensionale di numeri: uno scalare, un vettore, una matrice o un insieme di matrici. |
| **API (Application Programming Interface)** | Insieme delle funzioni pubbliche che modello e training engine possono chiamare. Nasconde i dettagli CPU e Metal. |
| **Shape** | Dimensioni di un tensore. Per esempio `[B,S,C]` indica batch, sequenza e caratteristiche. |
| **Rank** | Numero di dimensioni della shape. `[B,S,C]` ha rank 3. |
| **View** | Nuovo descrittore che interpreta gli stessi dati con una shape differente, senza duplicare il buffer. |
| **Reshape** | Cambia il modo in cui vengono interpretate le dimensioni senza cambiare o copiare i dati. |
| **Matmul** | Moltiplicazione fra matrici. E' l'operazione principale delle proiezioni lineari del Transformer. |
| **Transpose logica** | Legge una matrice scambiando righe e colonne senza creare necessariamente una nuova copia. Serve soprattutto nel backward. |
| **Riduzione** | Riassume molti valori in pochi risultati, per esempio calcolando somma, massimo o media dei quadrati. |
| **Gather rows** | Seleziona dalla tabella embedding le righe indicate dagli identificatori dei token. |
| **Scatter-add rows** | Operazione inversa usata nel backward: somma i gradienti nelle righe corrette della tabella embedding. |
| **Accumulate in-place** | Somma un tensore dentro un altro gia' esistente, senza creare un terzo buffer. Serve a combinare gradienti. |
| **In-place** | Operazione che modifica direttamente uno dei tensori ricevuti invece di scrivere in un output separato. |
| **Dtype** | Formato usato per conservare ogni numero del tensore. Determina precisione, memoria occupata e operazioni disponibili. |
| **F32 o FP32** | Numero floating-point a 32 bit. E' il formato iniziale del training perche' semplice e numericamente affidabile. |
| **F16 o FP16** | Floating-point a 16 bit. Occupa meno memoria ed e' spesso piu' veloce, ma ha un intervallo numerico minore. |
| **BF16** | Formato a 16 bit con intervallo simile a F32 ma meno precisione. E' spesso utile nel training mixed precision. |
| **U32** | Intero positivo a 32 bit. Nel runtime viene usato soprattutto per gli identificatori dei token. |
| **Mixed precision** | Training che combina formati a 16 e 32 bit per ottenere velocita' e stabilita'. Verra' dopo il primo training interamente F32. |
| **Loss scaling** | Tecnica del mixed precision che ingrandisce temporaneamente la loss per evitare che piccoli gradienti diventino zero in F16. |
| **CPU** | Processore general-purpose. Nel progetto esegue il riferimento leggibile e aiuta a verificare i risultati. |
| **GPU** | Processore progettato per molti calcoli paralleli. Su Apple Silicon viene usato per accelerare il training. |
| **Apple Silicon** | Famiglia di chip Apple in cui CPU e GPU condividono lo stesso sistema di memoria fisica. |
| **Metal** | API Apple usata per inviare buffer e calcoli alla GPU. Il modello non deve conoscere direttamente Metal. |
| **Backend** | Implementazione del runtime per un hardware specifico. CPU e Metal sono due backend con la stessa API pubblica. |
| **CUDA** | Tecnologia NVIDIA per eseguire calcoli su GPU. Non serve su Apple Silicon e non appartiene al Runtime v1. |
| **Kernel** | Piccola funzione numerica eseguita dalla CPU o dalla GPU su molti elementi. Riceve dati gia' validati e non conosce il modello. |
| **Fusion** | Unione di piu' operazioni in un solo kernel per evitare passaggi intermedi in memoria. E' un'ottimizzazione successiva. |
| **Provider** | Una possibile implementazione di una stessa operazione, per esempio CPU, Metal tile16 o Metal tile32. |
| **Dispatcher** | Componente che riceve una operazione gia' validata e sceglie il provider piu' conveniente per quella forma e situazione. |
| **Benchmark** | Misura ripetuta del tempo impiegato da una operazione. Serve a confrontare CPU, Metal e varianti di kernel sul Mac reale. |
| **Autotuning** | Processo che prova automaticamente piu' provider, verifica i risultati e conserva la scelta piu' efficiente. |
| **Profilo di tuning** | File che salva le scelte del benchmark per uno specifico chip, sistema operativo e versione del runtime. |
| **Cost model** | Stima del costo totale di una scelta, includendo kernel, sincronizzazione e posizione corrente dei dati. |
| **Operation signature** | Descrizione completa di una richiesta: operazione, dtype, shape, layout e opzioni. Viene usata come chiave della cache. |
| **Latenza** | Tempo necessario a completare una singola chiamata. E' importante soprattutto per operazioni piccole. |
| **Throughput** | Quantita' di lavoro completata in un certo tempo. E' importante per operazioni grandi e ripetute. |
| **Trace** | Registro leggibile delle decisioni del runtime, per esempio “questa matmul e' stata eseguita su Metal tile32”. |
| **Sincronizzazione** | Attesa necessaria affinche' i calcoli GPU pendenti siano terminati prima di leggere i risultati. Troppe sincronizzazioni rallentano il training. |
| **Batch di operazioni Metal** | Raggruppa piu' operazioni GPU prima di attendere. Riduce il costo di invio e sincronizzazione. Non coincide con il batch di esempi del training. |
| **Reference counting** | Contatore delle view che condividono lo stesso storage. Il buffer viene liberato soltanto quando nessuna view lo usa piu'. |
| **Storage** | Buffer che possiede realmente la memoria; il tensore descrive come interpretarla. |
| **Buffer** | Area di memoria che contiene i numeri di uno storage. Su Metal viene resa accessibile alla GPU. |
| **Alias** | Situazione in cui due tensori indicano la stessa area di memoria. Deve essere controllata per evitare sovrascritture impreviste. |
| **Broadcasting** | Ripetizione automatica di una dimensione piccola per combinarla con una piu' grande. Il v1 evita questa regola implicita. |
| **Row-major contiguo** | Layout in cui gli elementi consecutivi dell'ultima dimensione sono adiacenti in memoria. E' il layout semplice usato dal v1. |
| **SIMD** | Istruzioni CPU che applicano lo stesso calcolo a piu' numeri contemporaneamente. Sono un'ottimizzazione interna. |
| **KV cache** | Memoria delle Key e Value gia' calcolate durante la generazione. Evita di ricalcolare tutto il testo a ogni nuovo token; non serve per il primo training. |
| **Quantizzazione** | Uso di numeri a precisione molto bassa per ridurre memoria e accelerare il modello. Non appartiene al Runtime v1. |
| **Autograd** | Sistema che registra automaticamente le operazioni e costruisce il backward. Nel v1 il backward viene scritto esplicitamente per ogni layer. |

## 3. Confine del Runtime v1

Il runtime possiede:

- backend e sincronizzazione;
- storage e tensori;
- operazioni numeriche forward e backward;
- aggiornamento AdamW di un singolo parametro;
- validazione di forme, dtype, layout, alias e device;
- un dispatcher Apple limitato a provider CPU e Metal;
- caricamento delle decisioni prodotte dal benchmark sul Mac reale.

Il modello e il training engine possiedono:

- configurazione del Transformer;
- layer, parametri e gradienti;
- ordine del forward e del backward;
- elenco dei parametri da passare ad AdamW;
- micro-batch, dataset, checkpoint e logging;
- generazione e sampling.

Il Runtime v1 non include:

- autograd o un grafo dinamico;
- un sistema generico di plugin o provider per hardware arbitrario;
- un planner capace di riscrivere l'intero grafo del modello;
- CUDA o esecuzione distribuita;
- attention o SwiGLU fuse;
- KV cache ottimizzata;
- quantizzazione;
- mixed-precision training completo e loss scaling;
- trasferimenti automatici fra memorie fisicamente separate.

Queste funzionalita' possono essere aggiunte dopo il primo training end-to-end.

Il dispatcher Apple non e' un fallback nascosto. Esiste soltanto quando il
chiamante crea esplicitamente un backend Apple in modalita' automatica. I
backend CPU e Metal forzati continuano a eseguire esclusivamente sul device
richiesto.

## 4. Forma canonica dei dati

Per limitare view e permutazioni, il primo modello usa forme canoniche precise:

```text
token ids       [B, S]          U32
hidden state    [B, S, C]       F32
linear input    [B*S, C]        F32, tramite reshape senza copia
query           [B, S, Hq, D]   F32
key/value       [B, S, Hkv, D]  F32
attention out   [B, S, Hq, D]   F32
logits          [B*S, V]        F32
targets         [B*S]           U32
loss            []              F32
```

Significato delle lettere:

| Simbolo | Significato |
|---|---|
| `B` | numero di sequenze elaborate insieme nel batch |
| `S` | numero di token in ogni sequenza |
| `C` | dimensione dell'hidden state del modello |
| `Hq` | numero di query head |
| `Hkv` | numero di key/value head condivise dalla GQA |
| `D` | numero di valori contenuti in una singola attention head |
| `V` | numero totale di token nel vocabolario |
| `B*S` | tutti i token del batch trattati come righe di una matrice |
| `[]` | scalare, cioe' un solo numero senza dimensioni |

Vincoli:

- `C == Hq * D`;
- `Hq % Hkv == 0` per GQA;
- tensori materializzati row-major e contigui;
- `reshape` e' l'unica view obbligatoria nel primo milestone;
- Q, K e V usano proiezioni separate, quindi non serve inizialmente dividere un
  tensore QKV concatenato.

Questa scelta evita di rendere `transpose`, `permute`, slicing arbitrario e
broadcasting prerequisiti del training.

## 5. API pubblica minima

`include/runtime/runtime.h` resta l'umbrella pubblico compatibile. Tipi,
backend, tensori e operazioni sono divisi rispettivamente in `types.h`,
`backend.h`, `tensor.h` e `operations.h`. Le funzioni gia' presenti e
utilizzabili vengono conservate.

### 5.1 Backend

Gia' presenti:

```c
llm_status llm_backend_cpu_create(llm_backend **out_backend);
llm_status llm_backend_cpu_create_with_config(
    const llm_cpu_backend_config *config,
    llm_backend **out_backend);
llm_status llm_backend_metal_create(llm_backend **out_backend);
int llm_backend_metal_is_available(void);
void llm_backend_destroy(llm_backend *backend);
llm_device_type llm_backend_device(const llm_backend *backend);
llm_status llm_backend_synchronize(llm_backend *backend);
```

Da aggiungere:

`llm_device_type` acquisisce il valore `LLM_DEVICE_APPLE`, che identifica lo
storage condiviso gestito dal dispatcher automatico.

```c
typedef struct llm_apple_backend_config {
    int deterministic;
    int autotune;
    const char *tuning_profile_path;
} llm_apple_backend_config;

llm_status llm_backend_apple_create(
    const llm_apple_backend_config *config,
    llm_backend **out_backend);

llm_status llm_backend_begin_batch(llm_backend *backend);
llm_status llm_backend_end_batch(llm_backend *backend);
```

Esistono quindi tre modalita' chiare:

| Backend | Comportamento |
|---|---|
| CPU | forza sempre i kernel CPU; serve come riferimento e debug |
| Metal | forza sempre i kernel Metal; non puo' eseguire fallback CPU |
| Apple automatico | usa storage condiviso e sceglie CPU o Metal tramite il profilo di tuning |

Il backend Apple automatico alloca buffer Metal in memoria condivisa, accessibili
anche dai provider CPU. Registra chi ha scritto per ultimo ogni storage e
inserisce la sincronizzazione necessaria prima di cambiare provider. Non copia
silenziosamente il tensore in un secondo buffer.

Il batch generico e' importante su Metal: un training step contiene molte
operazioni e non deve creare una sincronizzazione GPU dopo ciascuna chiamata.
Su CPU mantiene l'ordine ed e' un no-op.

Dentro un batch Metal gia' aperto, il dispatcher mantiene le operazioni su
Metal: una CPU apparentemente piu' rapida per un piccolo kernel non giustifica
la sincronizzazione dell'intera catena GPU.

Le metriche Metal esistenti restano API diagnostiche e non vengono usate dal
modello.

### 5.2 Tensori e memoria

Gia' presenti:

```text
llm_tensor_create / destroy / move
llm_tensor_is_contiguous / device
llm_tensor_zero / fill_f32 / copy
llm_tensor_read / write
llm_cast
```

Implementato:

```c
llm_status llm_tensor_reshape(const llm_tensor *input,
                              size_t rank,
                              const size_t *shape,
                              llm_tensor *out_view);
```

`reshape` non copia i dati e richiede stesso `element_count`. Lo storage usa
reference counting: distruggere una view rilascia un riferimento, non
necessariamente il buffer.

Non sono richiesti ora:

```text
transpose
permute
slice arbitraria
copy strided
transfer tensor-to-tensor CPU ↔ Metal
```

I dati entrano ed escono dal backend scelto tramite `llm_tensor_write/read`, che
sono gia' trasferimenti host espliciti.

### 5.3 Primitive matematiche

Gia' presenti su CPU e nel codice Metal:

```text
add
multiply
scale
sum/max/mean-square sull'ultima dimensione
matmul F32 2D
matmul F16/BF16 → F32 2D
gather rows
scatter-add rows
softmax last
cross-entropy forward/backward
```

Da aggiungere:

```c
typedef struct llm_matmul_options {
    int transpose_left;
    int transpose_right;
} llm_matmul_options;

llm_status llm_matmul_ex(llm_backend *backend,
                         const llm_tensor *left,
                         const llm_tensor *right,
                         const llm_matmul_options *options,
                         llm_tensor *output);

llm_status llm_accumulate(llm_backend *backend,
                          const llm_tensor *source,
                          llm_tensor *destination);

```

`llm_matmul_ex` resta 2D nel v1. I flag di transpose permettono di esprimere i
gradienti delle proiezioni senza materializzare matrici trasposte.

`llm_accumulate` esegue `destination += source` ed e' esplicitamente in-place.
Serve per residual, parametri condivisi e accumulo fra micro-batch.

### 5.4 Operazioni Transformer

```c
llm_status llm_silu(llm_backend *backend,
                    const llm_tensor *input,
                    llm_tensor *output);

llm_status llm_silu_backward(llm_backend *backend,
                             const llm_tensor *input,
                             const llm_tensor *output_gradient,
                             llm_tensor *input_gradient);

llm_status llm_rms_norm(llm_backend *backend,
                        const llm_tensor *input,
                        const llm_tensor *weight,
                        float epsilon,
                        llm_tensor *output);

llm_status llm_rms_norm_backward(llm_backend *backend,
                                 const llm_tensor *input,
                                 const llm_tensor *weight,
                                 const llm_tensor *output_gradient,
                                 float epsilon,
                                 llm_tensor *input_gradient,
                                 llm_tensor *weight_gradient);

llm_status llm_rope(llm_backend *backend,
                    const llm_tensor *input,
                    const llm_tensor *cos_table,
                    const llm_tensor *sin_table,
                    size_t position_offset,
                    llm_tensor *output);

llm_status llm_rope_backward(llm_backend *backend,
                             const llm_tensor *output_gradient,
                             const llm_tensor *cos_table,
                             const llm_tensor *sin_table,
                             size_t position_offset,
                             llm_tensor *input_gradient);
```

RMSNorm, SiLU e RoPE sono operazioni pubbliche dirette. Non serve esportare
`add_scalar`, `rsqrt` o sigmoid: sono dettagli delle rispettive implementazioni.

### 5.5 Attention causale con GQA

Attention viene esposta come una singola operazione semantica. Questo evita di
richiedere subito transpose, batched matmul e materializzazione pubblica della
matrice `[S,S]`.

```c
typedef struct llm_attention_options {
    float scale;
    size_t query_position_offset;
} llm_attention_options;

llm_status llm_attention_forward(
    llm_backend *backend,
    const llm_tensor *query,
    const llm_tensor *key,
    const llm_tensor *value,
    const llm_attention_options *options,
    llm_tensor *output);

llm_status llm_attention_backward(
    llm_backend *backend,
    const llm_tensor *query,
    const llm_tensor *key,
    const llm_tensor *value,
    const llm_tensor *output_gradient,
    const llm_attention_options *options,
    llm_tensor *query_gradient,
    llm_tensor *key_gradient,
    llm_tensor *value_gradient);
```

Contratto:

- Q ha forma `[B,S,Hq,D]`;
- K e V hanno forma `[B,S,Hkv,D]`;
- `Hq % Hkv == 0`;
- la maschera e' sempre causale;
- scaling e softmax stabile usano FP32;
- backward puo' ricalcolare score e probabilita' invece di conservarli;
- nessun tensore viene trasferito alla CPU da un backend Metal.

Il primo kernel CPU puo' essere semplice e leggibile. Il primo percorso Metal
puo' essere composto o diretto, ma deve restare interamente GPU. Flash Attention
e fusion non appartengono al v1.

### 5.6 AdamW

```c
typedef struct llm_adamw_options {
    float learning_rate;
    float beta1;
    float beta2;
    float epsilon;
    float weight_decay;
    float gradient_scale;
    unsigned long long step;
} llm_adamw_options;

llm_status llm_adamw_update(llm_backend *backend,
                            llm_tensor *parameter,
                            const llm_tensor *gradient,
                            llm_tensor *first_moment,
                            llm_tensor *second_moment,
                            const llm_adamw_options *options);
```

Parameter, first moment e second moment vengono aggiornati in-place. Il training
engine crea e salva gli stati, determina lo step e passa `gradient_scale`, che
normalizza l'accumulo su micro-batch. In un incremento successivo potra'
includere anche il fattore di gradient clipping calcolato sul device.

Il primo training funzionante usa parametri, gradienti e stati AdamW in F32.
FP16/BF16 e loss scaling vengono aggiunti solo dopo la verifica end-to-end F32.

## 6. Come il modello usa il runtime

### 6.1 Forward di un decoder block

```text
input
 ├─ RMSNorm
 ├─ Wq/Wk/Wv con matmul
 ├─ reshape Q/K/V
 ├─ RoPE su Q e K
 ├─ causal GQA attention
 ├─ reshape + proiezione Wo
 └─ residual add

residual
 ├─ RMSNorm
 ├─ gate = SiLU(X × Wgate)
 ├─ up   = X × Wup
 ├─ hidden = gate × up
 ├─ output = hidden × Wdown
 └─ residual add
```

### 6.2 Backward

- matmul con transpose calcola gradienti di input e pesi;
- `llm_accumulate` combina i rami residuali;
- gli appositi backward coprono RMSNorm, SiLU, RoPE e attention;
- `llm_scatter_add_rows` calcola il gradiente dell'embedding;
- `llm_cross_entropy_backward` avvia il backward dai logits.

Non serve un sistema autograd per il primo modello: ogni layer conserva i
tensori necessari e implementa esplicitamente il proprio backward.

## 7. Layer interni minimi

```text
Modello e training engine
          ↓
API pubblica Runtime
          ↓
Validazione e dispatch comune
          ↓
Backend forzato oppure dispatcher Apple
          ↓
Provider CPU o Metal
          ↓
Kernel CPU oppure Metal
```

### API pubblica

Valida backend, tensori, forme, dtype, layout e alias. Definisce la semantica
comune e chiama il backend.

### Backend

L'attuale `llm_backend_ops` puo' essere esteso con le poche operazioni mancanti.

Il backend gestisce memoria, batch, sincronizzazione e scelta delle proprie
varianti. Metal puo' continuare a scegliere privatamente tile e pipeline; CPU
puo' continuare a scegliere privatamente thread e SIMD.

### Dispatcher Apple

Il dispatcher automatico e' intenzionalmente piccolo: conosce soltanto provider
CPU e Metal disponibili su Apple Silicon. La chiave di selezione contiene:

```text
operazione
dtype
shape
layout
modalita' deterministica
provider che ha scritto per ultimo gli input
batch Metal aperto o lavoro GPU pendente
```

Il risultato viene cercato nel profilo di tuning e conservato in cache. Ogni
decisione deve essere osservabile tramite metriche o trace, cosi' e' sempre
possibile capire dove una operazione e' stata eseguita.

La decisione non confronta soltanto il tempo del kernel. Usa il costo totale:

```text
costo totale = esecuzione + sincronizzazione + cambio provider
```

Questo evita il caso in cui una `add` piccola sembri piu' veloce su CPU ma
costringa ad attendere una lunga catena Metal, rendendo piu' lento l'intero
training step.

### Kernel

Riceve buffer e parametri gia' validati. Non conosce `llm_tensor`, modello o
layer, non alloca e non effettua fallback.

## 8. Benchmark e profilo di autotuning

Il benchmark viene costruito soltanto quando CPU e Metal implementano tutte le
operazioni v1 con la stessa semantica. Per ogni operazione misura almeno:

```text
provider CPU con diversi thread count
provider Metal e sue varianti compatibili
dtype
shape reali del modello
latenza di una chiamata isolata
throughput dentro una catena di operazioni
costo delle sincronizzazioni
memoria temporanea utilizzata
```

Prima di misurare la velocita', il benchmark confronta ogni risultato Metal con
CPU entro una tolleranza numerica dichiarata. Un provider scorretto viene
escluso anche se e' piu' veloce.

Il risultato e' un profilo legato a:

```text
modello del chip Apple
versione macOS e Metal
versione del runtime e dei kernel
modalita' deterministica
```

Se una di queste informazioni cambia, il profilo viene considerato non valido e
deve essere rigenerato. Il runtime non riutilizza misure prodotte da un altro
Mac come se fossero universali.

La chiave minima di una scelta salvata e':

```text
(operazione, dtype, shape, layout, stato di esecuzione) -> provider
```

`stato di esecuzione` distingue almeno una chiamata isolata da una operazione
inserita in un batch Metal. Il benchmark produce anche un confronto end-to-end
del training step: la somma dei migliori kernel isolati non garantisce infatti
la migliore esecuzione dell'intera rete.

Modalita' operative:

- senza profilo, il backend Apple usa una politica conservativa e osservabile;
- con autotuning abilitato, puo' misurare una nuova shape al primo utilizzo e
  conservarne la scelta;
- con profilo offline, carica le scelte gia' misurate prima del training;
- il chiamante puo' sempre forzare CPU o Metal per debug e confronto.

## 9. Stato reale

| Funzionalita' | CPU | Metal | Necessaria per il training v1 |
|---|:---:|:---:|:---:|
| lifecycle, memoria, sync | presente | presente | si |
| F32 e U32 | presente | presente | si |
| F16/BF16 storage e matmul | presente | presente | dopo F32 |
| add/multiply/scale | presente | presente | si |
| riduzioni last | presente | presente | si |
| matmul 2D | presente | presente | si |
| matmul con transpose logica | presente | assente | si |
| gather/scatter embedding | presente | presente | si |
| softmax | presente | presente | interna ad attention |
| cross-entropy forward/backward | presente | presente | si |
| storage reference counting | presente | presente | completato |
| reshape senza copia | presente | presente | completato |
| accumulate in-place | presente | assente | si |
| SiLU forward/backward | presente | assente | si |
| RMSNorm forward/backward | presente | assente | si |
| RoPE forward/backward | presente | assente | si |
| causal GQA attention forward/backward | presente | assente | si |
| AdamW | presente | assente | si |
| batch portabile | solo CPU sync | API Metal specifica | si per efficienza Metal |
| backend Apple automatico | assente | assente | si, dopo la parita' CPU/Metal |
| benchmark per operazione | presente per tutto il training v1 | presente per primitive correnti | Metal da estendere con i nuovi kernel |
| profilo e dispatcher autotuned | assente | assente | si, dopo il benchmark completo |
| KV cache | assente | assente | no, dopo il training |
| mixed-precision training | assente | assente | no, dopo F32 |
| provider/planner per hardware arbitrario | assente | assente | no |

La build Debug e tutti i 21 test CTest passano. I kernel CPU di training sono
coperti anche da AddressSanitizer e UndefinedBehaviorSanitizer. In questa esecuzione il device
Metal non era disponibile: il backend e i test Metal compilano, ma la parte GPU
e' stata saltata. La conformita' Metal deve essere eseguita su un processo che
veda realmente la GPU Apple.

## 10. Ordine di implementazione

### Milestone 1 — Congelare tutti i contratti

Prima di aggiungere kernel vengono definite tutte le operazioni necessarie:

- firma pubblica;
- formula matematica;
- shape e dtype;
- alias e comportamento in-place;
- NaN, infinito e codici di errore;
- forward, backward e tolleranza numerica;
- forme reali che il benchmark dovra' misurare.

L'uscita e' una suite di conformita' comune, inizialmente senza pretendere che
ogni test disponga gia' di entrambe le implementazioni.

### Milestone 2 — Implementazione CPU completa

**Stato: completata per il contratto di training F32.**

CPU implementa l'intero contratto v1:

- reference counting e reshape;
- matmul con transpose e accumulate;
- SiLU, RMSNorm, RoPE e causal GQA attention;
- tutti i backward;
- AdamW;
- test numerici e gradient check.

CPU non e' il backend scelto per addestrare il modello finale: e' il riferimento
semplice e deterministico contro cui verificare Metal.

### Milestone 3 — Implementazione Metal completa

Metal implementa le stesse operazioni e gli stessi errori:

- forward, backward e AdamW interamente GPU;
- batch generico e sincronizzazione;
- forme limite e forme non multiple dei tile;
- parita' con CPU entro tolleranze esplicite;
- nessun fallback CPU nel backend Metal forzato.

La milestone termina soltanto quando ogni riga del contratto ha una
implementazione CPU e una Metal verificata.

### Milestone 4 — Benchmark completo sul Mac reale

Per ogni operazione, dtype e famiglia di shape:

- misura CPU con diversi numeri di thread;
- misura tutte le varianti Metal compatibili;
- misura chiamate isolate e operazioni dentro una catena Metal;
- include sincronizzazione e temporanei nel costo;
- elimina i provider numericamente scorretti;
- salva il profilo associato all'hardware.

Viene poi misurato anche un decoder block completo, per controllare che le
decisioni locali migliorino davvero il workload reale.

### Milestone 5 — Dispatcher Apple automatico

- storage condiviso accessibile a CPU e Metal;
- cache delle decisioni per operation signature;
- tracking dell'ultimo provider che ha scritto ogni storage;
- sincronizzazione corretta quando cambia provider;
- trace che mostra ogni scelta;
- modalita' CPU, Metal e Apple automatico confrontabili.

Il dispatcher non cambia provider dentro una catena GPU soltanto perche' un
microbenchmark isolato favorisce CPU. La scelta segue il costo totale previsto.

### Milestone 6 — Training end-to-end su Apple Silicon

- training del tiny model con backend Metal forzato;
- training dello stesso modello con backend Apple automatico;
- loss finita e decrescente;
- gradienti e aggiornamenti confrontati con il riferimento CPU;
- nessuna chiamata `llm_tensor_read/write` durante lo step, salvo logging
  esplicito richiesto;
- confronto di tempo e memoria fra Metal e automatico;
- scelta predefinita basata sui risultati, non su un'ipotesi.

### Milestone 7 — Dopo il primo training

- KV cache e decode efficiente;
- mixed precision e loss scaling;
- norma globale e gradient clipping sul device;
- arena per temporanei;
- fusion e attention ottimizzata;
- eventuale estensione del dispatcher oltre Apple Silicon.

## 11. Criterio finale di completamento

Il Runtime v1 non e' completo soltanto perche' tutti i kernel hanno un test
unitario. Deve superare questo test di sistema:

```text
stesso tiny model
stessi pesi iniziali
stesso batch di token
        ↓
CPU riferimento:  forward → loss → backward → AdamW
Metal forzato:    forward → loss → backward → AdamW
Apple automatico: forward → loss → backward → AdamW
        ↓
loss finite e decrescente
gradienti e pesi entro tolleranze dichiarate
Metal forzato non esegue alcuna operazione sulla CPU
Apple automatico espone nel trace ogni provider scelto
benchmark end-to-end decide la modalita' predefinita
```

Questo e' il confine concreto: il runtime esiste per addestrare il modello sul
Mac e scegliere sulla base di misure reali, non per anticipare tutte le
astrazioni di un framework generale.
