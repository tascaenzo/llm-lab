# Backend Metal

**Stato (2026-08-17):** il codice Metal copre il contratto training v1,
inclusi RMSNorm, RoPE, attention causale GQA e AdamW. La suite contrattuale e
la parita' CPU/Metal restano gate obbligatori per ogni modifica; il profiling
per forma guida le ottimizzazioni, non la correttezza.
**Piattaforma:** macOS su Apple Silicon. Nessun fallback CPU.

## Confine

Il backend Metal implementa la stessa API pubblica e la stessa vtable del
backend CPU. Non deve aggiungere operazioni visibili soltanto a Metal. In
particolare cast, storage F16/BF16 e matmul mixed precision sono stati rimossi:
non sono richiesti dal training v1.

I soli dtype accettati sono F32 e U32.

## Infrastruttura presente

- selezione del device Metal predefinito;
- storage tramite buffer Metal;
- read/write, copy, fill e zero;
- command queue e sincronizzazione;
- batch asincrono esplicito begin/end;
- cache dei buffer e metriche diagnostiche;
- pipeline compilate da metallib o sorgente di fallback;
- primitive F32 gia' dichiarate dalla vtable.

Un backend Metal creato esplicitamente esegue soltanto pipeline Metal. Una
operazione non implementata restituisce `LLM_UNSUPPORTED_OPERATION`; non viene
inoltrata alla CPU.

## Primitive presenti

Metal copre lifecycle/memoria, batch asincroni, elementwise di base,
`accumulate`, riduzioni, gather/scatter-add, softmax, cross-entropy, SiLU,
RMSNorm, RoPE, attention causale GQA e AdamW, inclusi i backward richiesti dal
contratto. `matmul_ex` usa internamente `MPSMatrixMultiplication` per le
trasposizioni; l'API pubblica resta invariata.

La build Release e i test Metal sono stati eseguiti fuori dal sandbox su Apple
M4 reale: il test MPS `matmul_ex` seguito da un kernel Metal nello stesso batch
passa. Il benchmark GEMM F32 `512x512x512` ha misurato circa 360 GFLOP/s Metal
end-to-end (circa 0,745 ms) contro circa 95 GFLOP/s CPU a 10 thread. Forme
piccole restano piu' veloci su CPU per il costo di dispatch GPU.

I test Metal verificano queste primitive solo quando un device e' realmente
disponibile; su altre macchine compilazione e comportamento “unavailable”
restano verificabili.

## Gate di parita' e ottimizzazione

Ogni modifica Metal deve mantenere l'evidenza esecutiva su hardware:

1. esecuzione completa della suite contrattuale condivisa su Metal;
2. confronto esplicito CPU/Metal per output e gradienti del blocco minimal;
3. smoke training di almeno uno step senza fallback o trasferimenti intermedi;
4. benchmark e ottimizzazione sulle forme del primo modello.

Il trainer raggruppa un intero step Metal in un command buffer, evitando
sincronizzazioni tra forward, backward e AdamW; resta soltanto la lettura della
loss al termine dello step. Non aggiungere CUDA: l'API backend resta portabile,
ma senza hardware e CI CUDA non ci sarebbe una validazione affidabile della
parita' numerica.

## Risultato del Modello Minimal — sessione di training Metal

La sessione del 13 agosto 2026 ha addestrato su Metal il Modello Minimal con
`vocabulary_size=32001`, `hidden_size=64`, un layer, una head, contesto 32,
batch 2 e AdamW a learning rate fisso `0.001`. Il checkpoint a 2.000.000 step
e' versionato come artefatto di riferimento in
`artifacts/models/minimal-model/minimal-model-metal-step-2000000.llmckpt`
(SHA-256 `ce0957ca5aeeeb6960efc95a721d8539840904917737ccbf8147a876f6b0d803`).

La valutazione riproducibile sullo split validation, eseguita sul percorso CPU
di `model evaluate` (1.000 batch, seed 2026), misura `loss=3.78799526` e
`perplexity=44.16776639`. Sullo stesso campione, il checkpoint a 640.000 step
misura `loss=3.90422887` e `perplexity=49.61180814`: il training continua a
migliorare, pur con rendimenti decrescenti. Questo e' un risultato qualitativo
del modello; non sostituisce il confronto numerico CPU/Metal per operazione.

Ogni passo deve includere:

- pipeline e slot vtable, senza API backend-specifiche;
- shape non multiple dei tile;
- errori e alias validati dalla facciata comune;
- confronto CPU/Metal con tolleranza dichiarata;
- nessun `llm_tensor_read/write` host usato per calcolare il risultato.

## Attention e RoPE

Metal deve implementare esclusivamente la semantica training full-sequence:

```text
Q        [B,S,Hq,D]
K,V      [B,S,Hkv,D]
output   [B,S,Hq,D]
cos/sin  [S,D/2]
```

Attention e' sempre causale e `Hq % Hkv == 0`. Non esistono
`query_position_offset`, `position_offset`, sequenze Q/K di lunghezza
diversa o KV cache. Un futuro decode incrementale richiedera' un contratto e
una API separati.

## Batch e sincronizzazione

Fuori da un batch, una chiamata pubblica e' osservabile come sincrona. Dentro un
batch esplicito le pipeline possono essere accodate e `end_batch` attende il
command buffer. Errori di validazione devono essere rilevati prima di codificare
lavoro GPU.

Le metriche (dispatch, command buffer, tempi GPU, buffer attivi/cache) servono a
profilare il backend e non sono una dipendenza del modello.

## Correttezza numerica

La CPU e' il riferimento. L'uguaglianza bit-a-bit non e' richiesta per riduzioni
parallele o scatter-add con indici duplicati; la suite usa tolleranze per
operazione. NaN/Inf nelle opzioni vengono rifiutati dalla facciata comune.
Softmax, cross-entropy e AdamW devono rispettare gli stessi errori numerici CPU.

### Profilo per operazione

Un update di training e' una sequenza fissa e nota di operazioni, quindi il suo
profilo e' la somma di ogni forma misurata per il numero di volte che compare.
`utils/benchmarks/profile_model.py` enumera quelle forme con le molteplicita'
lette da `src/model` e le misura con `runtime_benchmark`:

```sh
python3 utils/benchmarks/profile_model.py \
  --benchmark build/release/utils/benchmarks/runtime_benchmark
```

Serve a scegliere cosa ottimizzare guardando i numeri invece del sorgente. La
prima esecuzione sul modello canonico ha corretto quattro priorita' che leggendo
il codice sembravano ovvie: l'output head e la cross-entropy, sospettati di
valere meta' dello step, pesano rispettivamente l'8,9% e l'1,6%, mentre
l'attention vale il 62,5% pur essendo il 4,1% delle operazioni aritmetiche.

Va rilanciato dopo ogni ottimizzazione: dice se il guadagno e' arrivato dove ci
si aspettava, e quando il collo di bottiglia si e' spostato altrove.

### Riproducibilita' per backend

Su Metal alcune riduzioni sommano con atomiche float in ordine non
deterministico: gradiente del peso di RMSNorm, gradienti di K e V
dell'attention, scatter-add dell'embedding e riduzione della loss. Il seed rende
quindi riproducibile la *sequenza dei batch*, non il valore esatto dei numeri: due
run Metal identici possono differire negli ultimi bit e divergere lentamente. La
build CPU resta deterministica a parita' di numero di thread. Chi ha bisogno di
un risultato bit-a-bit ripetibile deve usare la CPU o attendere riduzioni
deterministiche a due stadi.

## Gate di completamento

Metal e' pronto per il training solo quando:

- implementa tutti gli slot vtable del contratto v1;
- la suite condivisa passa su un device Apple reale;
- forward e backward di un piccolo blocco coincidono col riferimento CPU;
- un intero training step non fa fallback o round-trip host nascosti;
- sanitizer/CTest host e validazione Metal sono verdi;
- il benchmark misura solo F32/U32 e non riapre scope futuri.

Il runtime Metal e' parte del percorso di training v1; una modifica che non
supera questi gate non e' pronta per un run lungo.
