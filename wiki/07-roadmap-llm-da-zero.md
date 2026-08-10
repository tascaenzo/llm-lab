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

Il tokenizer Byte-level BPE e il corpus Wikipedia v1 sono ora implementati. Il
prossimo incremento della fase e' la pipeline descritta in
[dataset autoregressivo](08-dataset-autoregressivo.md): split per documento,
artefatti `.llmdat` e batch input/target.

## Fase 2 — Baseline semplice

- implementare un modello bigram o una piccola rete che predice il token successivo;
- addestrarlo e generare testo.

**Verifica:** la loss scende sensibilmente e i campioni riflettono il corpus.

## Fase 3 — Mini Transformer

- embedding di token e posizione;
- singola testa di attenzione causale;
- blocco MLP, residual e LayerNorm;
- piu' teste e piu' blocchi;
- head finale e cross-entropy.

**Verifica:** il modello supera la baseline e non puo' attendere token futuri (test della maschera causale).

## Fase 4 — Training affidabile

- AdamW, scheduler del learning rate e gradient clipping;
- validation periodica;
- checkpoint e ripresa del training;
- seed e logging delle metriche.

**Verifica:** un run interrotto riparte correttamente e la validation loss e' tracciata.

## Fase 5 — Generazione e analisi

- greedy, temperatura e top-k;
- prompt di confronto ripetibili;
- stima dei parametri, memoria e velocita'.

**Verifica:** lo stesso checkpoint produce output controllabili al variare della strategia di decoding.

## Fase 6 — Estensioni

- tokenizer BPE;
- dataset piu' ricco;
- mixed precision e GPU;
- KV cache;
- instruction fine-tuning su dati curati.

## Decisioni da prendere prima di iniziare

1. Quale hardware e' disponibile (solo CPU, GPU Apple, CUDA o cloud)?
2. Preferisci imparare con un corpus letterario italiano, documentazione tecnica o un dataset artificiale molto piccolo?
3. Vuoi privilegiare la massima semplicita' del codice o avvicinarci prima alle pratiche dei modelli moderni?

Quando sceglieremo queste tre cose, creeremo la struttura del progetto e partiremo dalla Fase 1.
