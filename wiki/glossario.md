# Glossario

| Termine | Definizione breve |
|---|---|
| Autoregressivo | Genera una sequenza prevedendo un token alla volta a partire dai precedenti. |
| Backend | Componente del runtime che gestisce un tipo di dispositivo e avvia i kernel adatti. |
| Batch | Gruppo di esempi elaborato insieme durante un singolo aggiornamento. |
| Checkpoint | Salvataggio dello stato del training, utile per riprendere o confrontare esperimenti. |
| Contesto | I token precedenti disponibili al modello quando predice il successivo. |
| Cross-entropy | Loss usata per misurare quanto la distribuzione prevista diverga dal target corretto. |
| Embedding | Vettore denso appreso che rappresenta un token. |
| Gradiente | Direzione e intensita' con cui modificare un parametro per ridurre la loss. |
| Kernel di calcolo | Implementazione concreta di un'operazione per CPU, Metal, CUDA o un altro backend. |
| Logit | Punteggio non normalizzato prodotto dal modello prima della softmax. |
| Parametri | Valori appresi dal modello, come pesi e bias. |
| Perplexity | `exp(cross-entropy)`; misura comparativa dell'incertezza predittiva. |
| Runtime tensoriale | Strato che rappresenta tensori, gestisce memoria ed esegue operazioni tramite un backend. |
| Softmax | Funzione che converte logits in probabilita' positive la cui somma e' 1. |
| SIMD | Istruzioni con cui un core applica la stessa operazione a piu' valori contemporaneamente. |
| Storage | Buffer di memoria che contiene i valori interpretati da un tensore. |
| Tensore | Insieme multidimensionale di numeri con forma, tipo e posizione in memoria. |
| Thread pool | Gruppo di thread persistenti usato per distribuire piu' lavori senza ricrearli a ogni operazione. |
| Tiling | Divisione di un calcolo in blocchi per riutilizzare meglio i dati nella cache. |
| Token | Unita' discreta di testo elaborata dal modello. |
| Transformer | Architettura basata sull'attenzione, usata dalla maggior parte dei LLM moderni. |
| Vocabolario | Insieme dei token ammessi dal tokenizer e rispettivi ID. |
