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

- aggiungere backward e registrazione delle operazioni;
- costruire embedding e layer lineari;
- implementare RMSNorm, RoPE e SwiGLU;
- verificare i gradienti numericamente.

**Verifica:** una piccola rete composta dai layer riduce una loss nota e ogni
gradiente coincide con una stima numerica entro la tolleranza dichiarata.

## Fase 5 — Transformer

- embedding dei token e posizione rappresentata con RoPE;
- singola testa di attenzione causale;
- blocco SwiGLU, connessioni residuali e RMSNorm;
- piu' teste e piu' blocchi;
- head finale e cross-entropy.

**Verifica:** il modello impara una piccola sequenza nota e cambiare un token
futuro non modifica le attivazioni delle posizioni precedenti.

## Fase 6 — Training affidabile

- AdamW, scheduler del learning rate e gradient clipping;
- validation periodica;
- checkpoint e ripresa del training;
- seed e logging delle metriche.

**Verifica:** un run interrotto riparte correttamente e la validation loss e' tracciata.

## Fase 7 — Generazione e analisi

- greedy, temperatura e top-k;
- prompt di confronto ripetibili;
- stima dei parametri, memoria e velocita'.

**Verifica:** lo stesso checkpoint produce output controllabili al variare della strategia di decoding.

## Fase 8 — Backend accelerati e scalabilita'

**Stato: backend Metal accelerato implementato e misurato; layer e fusion futuri.**

- backend Metal e CUDA;
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

Il runtime tensoriale CPU di riferimento e il backend Metal sono implementati.
Metal dispone di pool dei buffer, batch asincroni espliciti, metriche GPU,
riduzioni parallele, matmul tiled FP32/FP16/BF16 e benchmark riproducibile. Il
codice hardware-specifico resta isolato sotto `src/runtime/backends/`. Il
prossimo incremento funzionale puo' quindi costruire autograd e layer neurali
sopra l'API comune; fusion, autotuning persistente, CUDA e BLAS rimangono
ottimizzazioni o backend successivi.
