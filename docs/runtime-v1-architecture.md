# Runtime v1 — contratto minimo per il training

**Stato:** contratto CPU congelato; implementazione CPU completa; Metal da portare a parita'.
**Target:** macOS su Apple Silicon e Linux per il riferimento CPU. Windows non e' supportato.
**Dtype supportati:** F32 per dati numerici e U32 per indici. F16, BF16 e F8 sono
valori ABI riservati e devono produrre `LLM_UNSUPPORTED_DTYPE`.

## 1. Confine di responsabilita'

Il runtime possiede soltanto:

- backend, storage, tensori contigui e reshape senza copia;
- primitive numeriche necessarie al decoder Transformer v1;
- forward e backward espliciti;
- aggiornamento AdamW di un singolo parametro;
- validazione uniforme di dtype, shape, layout, device e alias.

Il modello e il training engine possiedono layer, ordine delle operazioni,
parametri, gradienti, micro-batch, checkpoint, dataset e logging.

Non appartengono al contratto v1: cast, mixed precision, matmul F16/BF16, KV
cache, autograd, broadcasting, slicing/permute generici, dispatcher automatico
CPU/Metal, fallback fra backend, quantizzazione, CUDA e Windows.

Ogni backend deve implementare esattamente la vtable corrente. Un backend
forzato non puo' trasferire il lavoro a un altro backend.

## 2. Dati canonici

```text
token ids       [B,S]          U32
hidden state    [B,S,C]        F32
linear input    [B*S,C]        F32, tramite reshape
query           [B,S,Hq,D]     F32
key/value       [B,S,Hkv,D]    F32
attention out   [B,S,Hq,D]     F32
logits          [B*S,V]        F32
targets         [B*S]          U32
loss            []             F32
```

Vincoli: `C == Hq*D`, `Hq % Hkv == 0`, rank massimo 4, dimensioni non
nulle, layout row-major contiguo. Tutti Q, K e V rappresentano la stessa
sequenza completa `S`.

## 3. Regole comuni

- Una funzione fallibile valida prima gli argomenti e non modifica output o
  stato se la validazione fallisce.
- Tensori di una stessa operazione devono appartenere allo stesso backend.
- Gli output devono essere gia' allocati, avere shape/dtype esatti e storage
  distinto dagli input, salvo le operazioni dichiarate in-place.
- `llm_accumulate`, `llm_scatter_add_rows` e `llm_adamw_update` sono le
  sole primitive numeriche pubbliche che modificano destinazioni esistenti.
- Gli indici fuori intervallo restituiscono `LLM_INVALID_INDEX`.
- Rank o dimensioni incompatibili restituiscono `LLM_INVALID_SHAPE`;
  alias vietato o opzioni invalide restituiscono `LLM_INVALID_ARGUMENT`.
- Layout non contiguo restituisce `LLM_UNSUPPORTED_LAYOUT`; device diverso
  restituisce `LLM_DEVICE_MISMATCH`.
- Le opzioni floating-point che devono essere positive e finite (`epsilon`,
  `attention.scale` e iperparametri AdamW) vengono rifiutate se NaN o Inf.
- Per i dati F32, NaN e Inf seguono l'aritmetica IEEE-754 della primitiva,
  eccetto softmax, cross-entropy e AdamW, che restituiscono
  `LLM_NUMERICAL_ERROR` quando non possono produrre un risultato finito
  valido secondo il proprio contratto.
- Il backend CPU e' sempre riproducibile a parita' di input, thread count e
  build; non esiste un'opzione pubblica per disattivare questa proprieta'.

## 4. Tensori e memoria

`llm_tensor_create` crea uno storage contiguo; rank zero indica uno scalare.
`llm_tensor_reshape` crea una view con lo stesso numero di elementi e reference
counting sullo storage. `move` trasferisce la proprieta'. `read` e `write`
richiedono l'intero payload; non esistono trasferimenti impliciti fra backend.

I soli dtype creabili sono F32 e U32. Mantenere i valori enum riservati evita
di cambiare l'ABI quando un futuro contratto introdurra' davvero la precisione
ridotta, senza obbligare i backend correnti a implementarla.

## 5. Contratti delle primitive

| Operazione | Formula e shape | Alias |
|---|---|---|
| add/multiply/scale | elementwise F32, shape identica | tutti distinti |
| reduce sum/max/mean-square | `[...,K] -> [...]` | distinti |
| matmul/ex | prodotto 2D, transpose logiche; dimensioni interne uguali | distinti |
| gather rows | table `[V,C]`, indices U32 shape `I`, output `I+[C]` | distinti |
| scatter-add rows | source `I+[C]`, indices `I`, table `[V,C]` | table in-place, storage distinti |
| accumulate | `destination += source`, shape identica | source e destination distinti |
| SiLU | `x*sigmoid(x)`; backward moltiplica per il gradiente a monte | distinti |
| RMSNorm | `y=x*w/sqrt(mean(x^2)+epsilon)` sull'ultima dimensione | distinti |
| RoPE | rotazione di coppie adiacenti su `[B,S,H,D]`; tabelle esatte `[S,D/2]` | tutti distinti |
| attention | full-sequence GQA causale, vedi sezione seguente | tutti distinti |
| softmax | stabile sull'ultima dimensione | distinti |
| cross-entropy | logits `[N,V]`, targets `[N]`, loss scalare; media su N | distinti |
| AdamW | aggiorna parameter, first moment e second moment F32 con stessa shape | tutti e quattro distinti |

I backward di SiLU, RMSNorm, RoPE e attention restituiscono gradienti con la
stessa shape dei rispettivi input. Il backward RMSNorm riduce il gradiente del
peso su tutte le righe.

## 6. Attention full-sequence

`llm_attention_options` contiene soltanto `scale`, finito e maggiore di zero.

- Q: `[B,S,Hq,D]`;
- K e V: `[B,S,Hkv,D]`;
- output: stessa shape di Q;
- `Hq % Hkv == 0`;
- Q, K e V devono avere lo stesso `B`, `S` e `D`;
- la query in posizione `i` vede soltanto le key `0..i`;
- score: `scale * dot(Q,K)`, softmax stabile in F32;
- il backward puo' ricalcolare score e probabilita'.

Decode incrementale, offset di posizione e KV cache non sono parte di questa
operazione. Saranno un contratto separato soltanto quando il runtime dovra'
supportare generazione efficiente.

## 7. RoPE full-sequence

RoPE accetta input o gradiente `[B,S,H,D]`, con D pari, e tabelle cosine e
sine entrambe esattamente `[S,D/2]`. Non accetta offset o tabelle piu' lunghe
della sequenza. Il backward applica la trasposta della stessa rotazione.

Questa scelta elimina due semantiche concorrenti dal runtime di training. Un
futuro decode incrementale potra' introdurre un'API distinta.

## 8. AdamW

L'API aggiorna un solo tensore parametro e i due momenti. L'orchestrazione su
una lista di parametri resta nel training engine. `step >= 1`,
`learning_rate > 0`, `epsilon > 0`, `gradient_scale > 0`,
`0 <= beta1,beta2 < 1` e weight decay finito.

Formula per elemento, con `g = gradient*gradient_scale`:

```text
m = beta1*m + (1-beta1)*g
v = beta2*v + (1-beta2)*g*g
p = p*(1-learning_rate*weight_decay)
    - learning_rate*(m/(1-beta1^step))/(sqrt(v/(1-beta2^step))+epsilon)
```

## 9. Backend CPU

Il backend CPU usa un pool pthread persistente. `thread_count=0` seleziona il
numero di CPU online; il chiamante partecipa al lavoro. Una chiamata
`parallel_for` annidata sullo stesso executor viene eseguita inline dal thread
corrente: non puo' attendere il pool che la sta gia' eseguendo e quindi non
crea deadlock.

Non esistono rami Win32, configurazioni di determinismo, cast o kernel a
precisione ridotta.

## 10. Requisiti per il backend Metal

Metal deve implementare la stessa vtable e superare la suite condivisa
`backend_contract_suite.c`. In particolare:

1. aggiungere matmul transpose e accumulate;
2. aggiungere SiLU e RMSNorm forward/backward;
3. aggiungere RoPE full-sequence forward/backward;
4. aggiungere attention GQA causale full-sequence forward/backward;
5. aggiungere AdamW;
6. mantenere ogni operazione sul device, senza fallback CPU;
7. verificare forme limite e dimensioni non multiple dei tile;
8. confrontare output e gradienti col riferimento CPU entro tolleranze
   dichiarate.

Le metriche e il batching Metal gia' pubblici sono diagnostica/esecuzione del
backend Metal; non ampliano la semantica matematica del runtime.

## 11. Gate di avanzamento

La CPU soddisfa il contratto quando:

- build Debug senza warning;
- test runtime, operazioni, executor e suite contrattuale passano con 1 e 4
  thread;
- AddressSanitizer e UndefinedBehaviorSanitizer non rilevano errori;
- benchmark smoke usa soltanto operazioni/dtype del contratto.

Si puo' iniziare Metal quando questi punti sono verdi e documenti/header
coincidono. Metal e' completo solo quando la stessa suite viene eseguita su un
device Apple reale e il training step non contiene read/write host intermedi.

Funzionalita' future richiedono prima una modifica esplicita di questo
documento, degli header e della suite condivisa; non devono apparire
preventivamente dentro un singolo backend.
