# Addestramento

## Obiettivo: prevedere il token successivo

Per ogni posizione, il modello produce una distribuzione di probabilita' sul vocabolario. La **cross-entropy loss** penalizza la probabilita' bassa assegnata al token corretto:

```text
loss = - media(log P(token_corretto | contesto))
```

Una loss piu' bassa indica previsioni migliori sul set valutato, ma non garantisce da sola risposte utili, fattuali o sicure.

## Il ciclo di training

```text
batch di token
  -> forward pass (logits)
  -> calcolo della loss
  -> backward pass (gradienti)
  -> optimizer.step() (aggiorna i pesi)
  -> azzera i gradienti
```

Il backward pass usa la backpropagation. L'ottimizzatore, spesso AdamW, usa i gradienti per modificare lentamente milioni o miliardi di parametri.

## Iperparametri fondamentali

| Iperparametro | Effetto |
|---|---|
| `d_model` | Larghezza delle rappresentazioni |
| `n_layers` | Profondita' del Transformer |
| `n_heads` | Numero di teste di attenzione |
| `context_length` | Numero massimo di token osservabili |
| `batch_size` | Esempi elaborati prima di un update |
| learning rate | Ampiezza degli aggiornamenti |
| numero di step | Durata effettiva del training |

La dimensione del modello e dei dati deve essere coerente con il budget di calcolo. Per il nostro LLM didattico sceglieremo parametri piccoli, in grado di funzionare anche su una macchina locale.

## Validazione e checkpoint

Il validation set non partecipa agli aggiornamenti: serve a osservare se il modello sta migliorando su dati non visti. Salveremo periodicamente un **checkpoint** con pesi del modello, stato dell'ottimizzatore, configurazione e step corrente; cosi' potremo riprendere il training o confrontare versioni diverse.

## Problemi tipici

- **Overfitting**: il modello memorizza il training set e generalizza male.
- **Instabilita'**: loss divergente, gradienti troppo grandi o learning rate eccessivo.
- **Memoria insufficiente**: l'attenzione cresce quadraticamente con la lunghezza del contesto.
- **Dati scadenti**: il modello impara rumore, errori e pattern indesiderati.

Prosegui con [inferenza e generazione](05-inferenza-e-generazione.md).
