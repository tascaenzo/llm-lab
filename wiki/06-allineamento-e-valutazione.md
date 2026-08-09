# Allineamento e valutazione

## Pretraining, fine-tuning, allineamento

Il pretraining insegna a continuare testo generale. Per trasformare questo modello in un assistente, di solito si aggiungono fasi ulteriori:

- **supervised fine-tuning (SFT)**: esempi prompt-risposta mostrano il formato desiderato;
- **preference optimization / RLHF**: preferenze umane o sintetiche spingono il modello verso risposte migliori;
- **safety tuning**: dati e valutazioni mirati riducono comportamenti dannosi.

Nel progetto didattico iniziale ci fermeremo al pretraining; l'SFT sara' un'estensione naturale quando il modello base funzionera'.

## Come valutare

| Cosa | Metodo |
|---|---|
| Capacita' predittiva | Cross-entropy e perplexity sul validation set |
| Apprendimento concreto | Prompt fissi campionati a ogni checkpoint |
| Generalizzazione | Test set separato e non contaminato |
| Robustezza | Casi limite, input vuoti, caratteri insoliti |
| Sicurezza e bias | Suite di test dedicata e revisione umana |

La **perplexity** e' approssimativamente `exp(loss)`: puo' essere letta come il numero effettivo di alternative tra cui il modello e' indeciso. E' utile per confrontare esperimenti sullo stesso tokenizer e sullo stesso set, non come misura universale della qualita' di un assistente.

## Principio operativo

Ogni esperimento dovrebbe registrare almeno configurazione, versione e dimensione del dataset, seed, curve di training/validation, checkpoint e alcuni output di esempio. Questo permette di capire *perche'* un cambiamento ha migliorato o peggiorato il modello.

Prosegui con la [roadmap di implementazione](07-roadmap-llm-da-zero.md).
