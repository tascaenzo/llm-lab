# Runtime tensoriale

Questo documento descrive la struttura del runtime v1. Il contratto normativo
delle operazioni e' in [runtime-v1-architecture.md](runtime-v1-architecture.md);
gli header pubblici restano la fonte di verita' compilabile.

## Obiettivo

Offrire al modello una API C sincrona e indipendente dal device, limitata alle
primitive necessarie al training F32 del decoder Transformer corrente. Il
modello non include header pthread o Metal e non conosce storage nativi.

Target supportati:

- CPU su macOS e Linux;
- Metal su macOS/Apple Silicon.

Windows, CUDA, mixed precision e fallback automatici non fanno parte del v1.

## Strati

```text
modello/training engine
        |
header pubblici runtime
        |
validazione comune
        |
vtable privata del backend
        |
CPU pthread oppure Metal
```

La validazione comune stabilisce shape, dtype, layout, device e alias prima di
entrare nel backend. I kernel ricevono buffer gia' validati e non aggiungono
semantiche proprie.

## Tipi e storage

`llm_tensor` descrive uno storage reference-counted, contiguo e row-major.
Rank massimo 4; rank zero e' uno scalare. Il descrittore non deve essere copiato
per assegnazione: si usano `llm_tensor_move` o `llm_tensor_reshape`.

`reshape` condivide lo storage e richiede lo stesso numero di elementi.
`read` e `write` trasferiscono sempre il payload completo fra host e backend.

I dtype del contratto sono:

- `LLM_DTYPE_F32`: valori, parametri, gradienti e stati ottimizzatore;
- `LLM_DTYPE_U32`: indici e target.

F16, BF16 e F8 restano enum riservati per stabilita' ABI, ma
`llm_tensor_create` deve rifiutarli con `LLM_UNSUPPORTED_DTYPE` su ogni
backend v1.

## API backend

La CPU espone creazione semplice o `llm_cpu_backend_config{thread_count}`.
Zero significa selezione automatica. L'esecuzione e' sincrona e riproducibile;
non esiste un flag `deterministic`.

Metal espone disponibilita', creazione, sincronizzazione, batch esplicito e
metriche diagnostiche. Un backend Metal forzato non puo' eseguire fallback CPU.

Ogni tensore deve essere distrutto prima del backend che lo possiede.

## API tensoriali

Sono incluse soltanto:

- create, destroy, move e reshape;
- zero, fill F32 e copy;
- read e write completi;
- device e verifica di contiguita'.

Non sono incluse cast, view strided, slice, permute, transpose materiale,
broadcasting o trasferimenti impliciti fra backend.

## Primitive numeriche

Il runtime pubblico contiene:

- add, multiply, scale e accumulate;
- reduce sum/max/mean-square sull'ultima dimensione;
- matmul 2D F32, anche con transpose logiche;
- gather e scatter-add di righe;
- SiLU e backward;
- RMSNorm e backward;
- RoPE full-sequence e backward;
- attention GQA causale full-sequence e backward;
- softmax;
- cross-entropy forward/backward;
- aggiornamento AdamW di un parametro.

Non contiene `llm_cast` o `llm_matmul_mixed_f32`. Aggiungerli in un singolo
backend senza una revisione comune del contratto sarebbe un errore di scope.

## Shape Transformer canoniche

```text
hidden       [B,S,C]
Q            [B,S,Hq,D]
K,V          [B,S,Hkv,D]
RoPE tables  [S,D/2]
logits       [N,V]
targets      [N]
loss         []
```

Attention richiede una sola lunghezza S per Q, K e V ed e' sempre causale.
RoPE richiede tabelle della lunghezza esatta S. Offset, decode incrementale e
KV cache sono volutamente assenti.

## Errori

- puntatori, opzioni o alias invalidi: `LLM_INVALID_ARGUMENT`;
- shape/rank incompatibili: `LLM_INVALID_SHAPE`;
- dtype non v1: `LLM_UNSUPPORTED_DTYPE`;
- layout non contiguo: `LLM_UNSUPPORTED_LAYOUT`;
- tensore di un altro backend: `LLM_DEVICE_MISMATCH`;
- indice fuori tabella: `LLM_INVALID_INDEX`;
- overflow nel calcolo delle dimensioni: `LLM_OVERFLOW`;
- risultato numericamente non valido dove vietato: `LLM_NUMERICAL_ERROR`.

La validazione deve avvenire prima di modificare output o stato.

## Estensione del runtime

Una nuova funzionalita' entra nel runtime soltanto se il modello/training engine
corrente la richiede. L'ordine obbligatorio e':

1. modifica del contratto documentale;
2. modifica degli header pubblici e della vtable comune;
3. casi positivi e negativi nella suite contrattuale;
4. implementazione CPU di riferimento;
5. implementazione e confronto Metal;
6. benchmark solo dopo la correttezza.

Questo impedisce ai backend di divergere o di accumulare API speculative.
