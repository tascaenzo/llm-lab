# Panoramica: cos'e' un LLM

Un LLM e' un modello statistico che, dato un contesto di token, stima quale token sia piu' probabile dopo di esso.

Formalmente, per una sequenza `x_1, x_2, ..., x_T`, un modello autoregressivo apprende:

```text
P(x_1, ..., x_T) = prodotto per t da 1 a T di P(x_t | x_<t)
```

Durante la generazione, applica la stessa regola ripetutamente: predice un token, lo aggiunge al contesto, poi predice il successivo.

## I componenti principali

| Componente | Responsabilita' | Input -> output |
|---|---|---|
| Corpus | Fornisce esempi linguistici | testi grezzi -> esempi puliti |
| Tokenizer | Divide e codifica il testo | testo -> ID interi |
| Embedding | Trasforma ID in vettori | ID -> vettori continui |
| Positional encoding | Indica l'ordine dei token | posizione -> vettori/offset |
| Transformer | Integra il contesto | vettori -> rappresentazioni contestuali |
| Language-model head | Calcola i punteggi per il prossimo token | stato -> logits sul vocabolario |
| Funzione di perdita | Misura l'errore di previsione | logits + target -> loss |
| Ottimizzatore | Aggiorna i pesi | gradienti -> nuovi pesi |
| Decoder | Sceglie i token in output | logits -> token -> testo |

## Due fasi da non confondere

### Addestramento

Il modello vede enormi quantita' di testo e tenta di indovinare il token seguente. Quando sbaglia, il calcolo dei gradienti modifica i suoi parametri. Questo e' il momento in cui apprende.

### Inferenza

I parametri rimangono fissi. Il modello riceve un prompt e genera token uno alla volta. Questo e' il momento in cui lo utilizziamo.

## Cosa “sa” un LLM

Un LLM non conserva una banca dati perfetta di fatti ne' ragiona con regole esplicite. Codifica regolarita' statistiche dei dati nei propri pesi. Per questo puo' produrre testo coerente e fare generalizzazioni, ma puo' anche allucinare informazioni, riflettere bias presenti nei dati o fallire fuori distribuzione.

Prosegui con [dati e tokenizzazione](02-dati-e-tokenizzazione.md).
