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
9. [Glossario](glossario.md): termini chiave.

## Limite del primo progetto

La prima implementazione non cerchera' di replicare un modello commerciale: costruiremo un **piccolo language model autoregressivo** addestrato su un corpus limitato. L'obiettivo e' capire ogni passaggio, osservare risultati misurabili e avere una base estendibile.

## Flusso essenziale

```text
testi -> tokenizer -> ID dei token -> embedding + posizioni
      -> blocchi Transformer -> distribuzione sul vocabolario
      -> token successivo -> testo generato
```
