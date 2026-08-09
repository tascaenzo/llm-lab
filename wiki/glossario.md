# Glossario

| Termine | Definizione breve |
|---|---|
| Autoregressivo | Genera una sequenza prevedendo un token alla volta a partire dai precedenti. |
| Batch | Gruppo di esempi elaborato insieme durante un singolo aggiornamento. |
| Checkpoint | Salvataggio dello stato del training, utile per riprendere o confrontare esperimenti. |
| Contesto | I token precedenti disponibili al modello quando predice il successivo. |
| Cross-entropy | Loss usata per misurare quanto la distribuzione prevista diverga dal target corretto. |
| Embedding | Vettore denso appreso che rappresenta un token. |
| Gradiente | Direzione e intensita' con cui modificare un parametro per ridurre la loss. |
| Logit | Punteggio non normalizzato prodotto dal modello prima della softmax. |
| Parametri | Valori appresi dal modello, come pesi e bias. |
| Perplexity | `exp(cross-entropy)`; misura comparativa dell'incertezza predittiva. |
| Softmax | Funzione che converte logits in probabilita' positive la cui somma e' 1. |
| Token | Unita' discreta di testo elaborata dal modello. |
| Transformer | Architettura basata sull'attenzione, usata dalla maggior parte dei LLM moderni. |
| Vocabolario | Insieme dei token ammessi dal tokenizer e rispettivi ID. |
