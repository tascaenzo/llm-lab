# Wiki: costruire e capire un LLM

Questa wiki spiega **quali elementi compongono un Large Language Model (LLM)**, come collaborano tra loro e quale percorso seguire per costruirne una versione didattica da zero.

Un LLM non e' soltanto una rete neurale: e' un sistema formato da dati, rappresentazioni numeriche, architettura Transformer, procedura di addestramento e metodo di generazione.

## Mappa di lettura

1. [01 — Panoramica](01-panoramica.md): il quadro completo e il flusso di un LLM.
2. [02 — Dati e tokenizzazione](02-dati-e-tokenizzazione.md): come il testo diventa numeri.
3. [03 — Architettura Transformer](03-architettura-transformer.md): il modello che predice il token successivo.
4. [04 — Addestramento](04-addestramento.md): come il modello apprende dai dati.
5. [05 — Inferenza e generazione](05-inferenza-e-generazione.md): come una previsione diventa una risposta.
6. [06 — Allineamento e valutazione](06-allineamento-e-valutazione.md): come rendere il modello utile e misurarne la qualita'.
7. [07 — Roadmap: LLM didattico da zero](07-roadmap-llm-da-zero.md): il progetto che realizzeremo dopo la teoria.
8. [08 — Dataset autoregressivo](08-dataset-autoregressivo.md): come i documenti diventano input e target.
9. [09 — Flusso dati e artefatti](09-flusso-dati-e-artefatti.md): guida completa ai file prodotti e al loro uso nel training.
10. [10 — Runtime tensoriale](10-runtime-tensoriale.md): come i calcoli del modello vengono eseguiti su CPU e, in futuro, GPU.
11. [11 — Backend CPU](11-backend-cpu.md): come usare thread, core, SIMD e cache conservando kernel leggibili.
12. [12 — Backend Metal](12-backend-metal.md): come gli stessi tensori diventano buffer e kernel eseguiti dalla GPU Apple.
13. [13 — Modello Minimal](13-primo-modello-addestrabile.md): la prima rete reale, piccola e verificabile su CPU.
14. [14 — Italiano-Base-75M](14-italiano-base-75m.md): il primo decoder pensato per completamenti italiani utili.
15. [15 — Esperimento Italiano-Chat-75M](15-esperimento-sft-75m-post-mortem.md): post-mortem del primo SFT, problemi emersi e piano verso il 300M.
16. [Glossario](glossario.md): termini chiave.

## Scala iniziale

La prima configurazione addestrabile sara' necessariamente piccola, cosi' da
poter verificare ogni componente su hardware locale. Runtime, backend e modello
saranno pero' progettati come componenti estendibili: ridurre le dimensioni del
primo esperimento non deve introdurre limiti strutturali nel codice.

## Flusso essenziale

```text
testi -> tokenizer -> ID dei token -> embedding + posizioni
      -> blocchi Transformer -> distribuzione sul vocabolario
      -> token successivo -> testo generato
```
