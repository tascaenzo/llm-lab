# Roadmap: LLM didattico da zero

Questa e' la sequenza proposta per trasformare la wiki in un progetto funzionante. Ogni fase ha un risultato verificabile prima della successiva.

## Fase 0 — Ambiente e obiettivo

- usare C23 e una toolchain CMake multipiattaforma;
- fissare un dataset con licenza chiara;
- decidere se usare CPU, GPU locale o cloud;
- definire un obiettivo ristretto: completare testo italiano su un corpus piccolo.

**Verifica:** la CLI stampa versione e configurazione; build e test sono riproducibili.

La toolchain e' descritta in [../docs/TOOLCHAIN.md](../docs/TOOLCHAIN.md); l'obiettivo didattico invariato e' implementare esplicitamente le strutture numeriche e il tokenizer.

## Fase 1 — Dati e tokenizer minimo

- caricare, pulire e separare il corpus;
- implementare `encode` e `decode` a caratteri o byte;
- creare batch input/target a token spostati.

**Verifica:** `decode(encode(testo))` funziona e un batch ha le forme attese.

Il tokenizer Byte-level BPE, il corpus Wikipedia v1 e la pipeline descritta in
[dataset autoregressivo](08-dataset-autoregressivo.md) sono implementati: split
per documento, artefatti `.llmdat` e batch input/target.

## Fase 2 — Runtime tensoriale

- rappresentare tensori, forma, dtype e dispositivo;
- gestire memoria e ownership;
- definire una API di operazioni indipendente dall'hardware;
- implementare backend e kernel CPU di riferimento;
- verificare ogni operazione con risultati noti.

La teoria e' in [runtime tensoriale](10-runtime-tensoriale.md); la specifica
concreta e' in
[docs/runtime-tensoriale.md](../docs/runtime-tensoriale.md).

**Verifica:** tensori, memoria e operazioni CPU superano test indipendenti dal
modello e non espongono dettagli hardware al chiamante.

## Fase 3 — Backend CPU parallelo

**Stato: implementato nella versione iniziale.**

- separare il codice specifico CPU in una directory di backend;
- creare un executor con thread pool persistente;
- parallelizzare elementwise, riduzioni, matmul e primitive per il modello;
- conservare i kernel C di riferimento;
- misurare scalabilita', tiling e primi percorsi SIMD.

La teoria e' in [backend CPU](11-backend-cpu.md); il progetto tecnico e' in
[docs/backend-cpu.md](../docs/backend-cpu.md).

**Verifica:** gli stessi kernel coincidono entro tolleranza con il riferimento
usando uno o piu' thread, non presentano race e migliorano in benchmark su forme
rappresentative.

## Fase 4 — Core neurale

**Stato: completata con il Modello Minimal.**

- costruire config, parametri, gradienti e inizializzazione riproducibile;
- implementare `embedding -> linear head -> logits -> cross-entropy`;
- calcolare backward di embedding e layer lineare con le primitive runtime;
- mantenere un registry di parametri e aggiornare tutti i pesi con AdamW;
- fare overfit di una fixture minuscola e verificare i gradienti numericamente.

Il risultato e' il Modello Minimal: un language model autoregressivo
addestrabile, non un mock. Config, checkpoint e API restano estendibili.

La CPU e' il riferimento per questo gate perche' soddisfa l'intero contratto
runtime v1. Il modello non deve dipendere dalla CPU: usera' la stessa API
backend-agnostic che in futuro permettera' di eseguire lo stesso test su Metal.
La guida operativa e' in [primo modello addestrabile](13-primo-modello-addestrabile.md).

**Verifica:** con seed fisso, gradienti di embedding/proiezione coincidono con
le differenze finite, un corpus minuscolo viene overfittato con loss in calo e
un run ripreso da checkpoint coincide con lo stesso run continuo.

## Fase 5 — Transformer

**Stato: il blocco causale e' nel Modello Minimal; scalabilita' generica ancora da fare.**

- embedding dei token e posizione rappresentata con RoPE;
- singola testa di attenzione causale;
- blocco SwiGLU, connessioni residuali e RMSNorm;
- piu' teste e piu' blocchi;
- head finale e cross-entropy.

**Verifica:** il modello impara una piccola sequenza nota e cambiare un token
futuro non modifica le attivazioni delle posizioni precedenti.

Il Modello Minimal implementa un blocco causale CPU con RMSNorm, Q/K/V, RoPE, attention,
proiezione e residual. Per mantenere il riferimento numerico semplice accetta
solo un layer e una head, senza MLP. Dopo la parita' Metal, lo stesso modello
verra' generalizzato a piu' layer, multi-head e SwiGLU; non sara' un secondo
modello indipendente.

## Fase 6 — Training affidabile

- scheduler del learning rate e gradient clipping;
- validation periodica;
- seed e logging delle metriche.

**Verifica:** un run interrotto riparte correttamente e la validation loss e' tracciata.

## Fase 7 — Generazione e analisi

- greedy, temperatura e top-k;
- prompt di confronto ripetibili;
- stima dei parametri, memoria e velocita'.

**Verifica:** lo stesso checkpoint produce output controllabili al variare della strategia di decoding.

## Fase 8 — Backend accelerati e scalabilita'

**Stato: accelerazione Metal di base validata; il prossimo gate e' la parita'
di training del Modello Minimal.**

- completare Metal sulle primitive richieste dal modello reale;
- test di conformita' tra dispositivi;
- mixed precision;
- kernel fusi e profiling.

**Verifica:** lo stesso modello produce risultati numericamente compatibili su
backend diversi senza modificare i layer.

## Fase 9 — Estensioni

- dataset piu' ricco;
- Multi-head Latent Attention;
- Mixture of Experts;
- Multi-Token Prediction;
- KV cache;
- attenzione sparsa per contesti lunghi;
- instruction fine-tuning su dati curati.

## Decisione corrente

Il runtime tensoriale CPU di riferimento implementa l'intero contratto di
training F32/U32. Metal ha una base validata su Apple M4: memoria, batch, GEMM
(anche trasposto via MPS), embedding, cross-entropy, `accumulate` e SiLU sono
disponibili; RMSNorm, RoPE, attention GQA e AdamW restano da implementare.

La decisione corrente e' fermare i run CPU lunghi dopo avere validato il
Modello Minimal e
portare a Metal l'intero training step: RMSNorm, RoPE, attention causale,
backward e AdamW. CPU e' il riferimento numerico per confrontare logits, loss,
gradienti e parametri aggiornati. Solo dopo questa parita' verranno avviati
run estesi e la configurazione multi-layer. CUDA resta fuori dallo scope finche' non esistono
hardware e CI per testarne correttezza e prestazioni. Mixed precision, KV cache,
fusion e backend ulteriori richiederanno contratti separati quando necessari.
