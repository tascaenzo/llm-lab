# Inferenza e generazione

In inferenza non cambiamo i pesi. Partiamo dal prompt, lo tokenizziamo e ripetiamo il ciclo seguente:

```text
contesto -> modello -> logits dell'ultima posizione -> scelta del token
         -> aggiunta al contesto -> ripetizione
```

## Strategie di scelta del token

| Strategia | Comportamento |
|---|---|
| Greedy decoding | Sceglie sempre il token piu' probabile; e' stabile ma puo' essere ripetitivo |
| Temperatura | Ridimensiona i logits; bassa = piu' prudente, alta = piu' varia |
| Top-k | Estrae solo dai `k` token piu' probabili |
| Top-p (nucleus) | Estrae dal piu' piccolo insieme con probabilita' cumulata almeno `p` |

La temperatura agisce prima della softmax:

```text
P(token) = softmax(logits / temperatura)
```

Con temperatura vicina a zero, il comportamento si avvicina alla scelta greedy. Valori alti aumentano la varieta' ma anche il rischio di testo incoerente.

## Fine e limiti della generazione

La generazione termina quando si produce `EOS`, si raggiunge un massimo di token o interviene una regola applicativa. Per un primo modello dovremo impostare limiti piccoli e stampare sia token sia testo decodificato, per rendere il processo osservabile.

## KV cache

Durante la generazione autoregressiva, ricalcolare attenzione su tutto il prompt a ogni token e' inefficiente. Una **KV cache** conserva le key e value calcolate nei blocchi precedenti; in questo modo il modello calcola soltanto il nuovo token. La aggiungeremo dopo aver ottenuto una versione semplice e corretta.

Prosegui con [allineamento e valutazione](06-allineamento-e-valutazione.md).
