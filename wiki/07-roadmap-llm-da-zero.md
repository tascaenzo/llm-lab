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

## Fase 5 — Decoder Transformer scalabile

**Stato: il blocco causale e' nel Modello Minimal; il prossimo target e'
Italiano-Base-75M.**

- embedding dei token e posizione rappresentata con RoPE;
- singola testa di attenzione causale;
- blocco SwiGLU, connessioni residuali e RMSNorm;
- piu' teste e piu' blocchi;
- head finale e cross-entropy.

**Verifica:** il modello impara una piccola sequenza nota e cambiare un token
futuro non modifica le attivazioni delle posizioni precedenti.

Il Modello Minimal implementa un blocco causale con RMSNorm, Q/K/V, RoPE,
attention, proiezione e residual. Per mantenere il riferimento numerico
semplice accetta solo un layer e una head, senza MLP. Lo stesso modello verra'
generalizzato a piu' layer, multi-head e SwiGLU; non sara' un secondo modello
indipendente. Il primo target concreto e' [Italiano-Base-75M](14-italiano-base-75m.md):
12 layer, hidden size 512, 8 head, SwiGLU 1536 e contesto 512.

## Fase 6 — Training affidabile e run Base-75M

- sampler streaming senza rimpiazzo ed epoche riproducibili: completato;
- scheduler del learning rate, gradient clipping, gradient accumulation e
  checkpoint periodici: completati;
- validation periodica, checkpoint migliore;
- seed, token elaborati e logging di metriche e memoria.

**Verifica:** un run interrotto riparte correttamente, la validation loss e'
tracciata e un'intera epoca sul corpus equivale a un numero noto di token.

## Fase 7 — Generazione e analisi

- greedy, temperatura e top-k;
- prompt di confronto ripetibili;
- stima dei parametri, memoria e velocita'.

**Verifica:** lo stesso checkpoint produce output controllabili al variare della strategia di decoding.

## Fase 8 — Backend accelerati e scalabilita'

**Stato: il training del Modello Minimal e' eseguito su Metal; il prossimo gate
e' la parita' esplicita del decoder scalabile.**

- completare Metal sulle primitive richieste dal modello reale;
- test di conformita' tra dispositivi;
- benchmark sulle forme 75M e profiling della memoria;
- kernel fusi e profiling.

**Verifica:** lo stesso modello produce risultati numericamente compatibili su
backend diversi senza modificare i layer.

## Fase 9 — Italiano-Chat-75M ed estensioni

- dataset istruzione/risposta italiano con licenza verificata;
- token di ruolo e loss mascherata per fine-tuning supervisionato;
- Multi-head Latent Attention;
- Mixture of Experts;
- Multi-Token Prediction;
- KV cache;
- attenzione sparsa per contesti lunghi;
- instruction fine-tuning su dati curati.

## Decisione corrente

Il runtime tensoriale CPU di riferimento implementa l'intero contratto di
training F32/U32. Metal implementa il percorso completo del Modello Minimal,
inclusi RMSNorm, RoPE, attention causale e AdamW; il checkpoint a 2 milioni di
step e' il riferimento integrato. CPU resta il riferimento numerico per
confrontare logits, loss, gradienti e parametri aggiornati.

La decisione corrente e' implementare il decoder scalabile e il trainer
affidabile richiesti da [Italiano-Base-75M](14-italiano-base-75m.md), poi
eseguire il primo run esteso sul Mac M4. CUDA resta fuori dallo scope finche'
non esistono hardware e CI per testarne correttezza e prestazioni. Mixed
precision, KV cache, fusion e backend ulteriori richiederanno contratti
separati quando necessari.
