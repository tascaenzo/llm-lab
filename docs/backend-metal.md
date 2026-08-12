# Backend Metal

**Stato (2026-08-12):** accelerazione di base validata su Apple M4; parita'
col contratto training v1 ancora da implementare. Il lavoro Metal e' in pausa
mentre viene costruito il primo modello CPU, cosi' le prossime ottimizzazioni
saranno guidate dalle forme e dai colli di bottiglia del modello reale.
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

Attualmente Metal copre lifecycle/memoria, batch asincroni, elementwise di
base, `accumulate`, riduzioni, gather/scatter-add, softmax, cross-entropy,
SiLU forward/backward e GEMM F32. `matmul_ex` usa internamente
`MPSMatrixMultiplication` per le trasposizioni; l'API pubblica resta invariata.

La build Release e i test Metal sono stati eseguiti fuori dal sandbox su Apple
M4 reale: il test MPS `matmul_ex` seguito da un kernel Metal nello stesso batch
passa. Il benchmark GEMM F32 `512x512x512` ha misurato circa 360 GFLOP/s Metal
end-to-end (circa 0,745 ms) contro circa 95 GFLOP/s CPU a 10 thread. Forme
piccole restano piu' veloci su CPU per il costo di dispatch GPU.

I test Metal verificano queste primitive solo quando un device e' realmente
disponibile; su altre macchine compilazione e comportamento “unavailable”
restano verificabili.

## Lavoro rimandato per la parita'

Prima di dichiarare Metal un backend di training completo restano:

1. RMSNorm forward/backward;
2. RoPE full-sequence forward/backward con tabelle `[S,D/2]`;
3. attention GQA causale full-sequence forward/backward con Q/K/V della stessa S;
4. AdamW F32;
5. esecuzione completa della suite contrattuale condivisa su Metal;
6. benchmark e ottimizzazione sulle forme del primo modello.

`accumulate`, matmul transpose e SiLU sono completati. Durante la pausa non
aggiungere CUDA: l'API backend resta portabile, ma senza hardware e CI CUDA non
ci sarebbe una validazione affidabile della parita' numerica.

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

## Gate di completamento

Metal e' pronto per il training solo quando:

- implementa tutti gli slot vtable del contratto v1;
- la suite condivisa passa su un device Apple reale;
- forward e backward di un piccolo blocco coincidono col riferimento CPU;
- un intero training step non fa fallback o round-trip host nascosti;
- sanitizer/CTest host e validazione Metal sono verdi;
- il benchmark misura solo F32/U32 e non riapre scope futuri.

Fino a quel punto e' corretto iniziare lo sviluppo Metal, ma non dichiarare il
runtime Metal completo.
