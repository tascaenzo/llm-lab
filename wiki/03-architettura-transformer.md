# Architettura Transformer

Per il nostro progetto useremo un **decoder-only Transformer**, la famiglia di architetture usata dai language model autoregressivi.

## Dal token al vettore

Ogni ID viene cercato in una matrice di embedding `E` di forma `vocab_size x d_model`. Il risultato e' un vettore di dimensione `d_model`.

Poiche' l'attenzione da sola non conosce l'ordine, sommiamo un'informazione di posizione:

```text
input al primo blocco = token_embedding + positional_embedding
```

## Il blocco Transformer

Un modello contiene `N` blocchi uguali in sequenza. Ciascun blocco ha due sottolivelli principali e connessioni residue.

```text
x -> LayerNorm -> attenzione causale multi-head -> + residuo
  -> LayerNorm -> MLP feed-forward              -> + residuo
```

### Self-attention causale

Ogni token costruisce tre vettori tramite proiezioni apprese:

- **Query (Q)**: cosa sta cercando questo token;
- **Key (K)**: quale informazione offre un token;
- **Value (V)**: il contenuto da trasferire.

L'attenzione e' calcolata come:

```text
Attention(Q, K, V) = softmax((Q K^T) / sqrt(d_k) + mask) V
```

La maschera causale assegna probabilita' nulla alle posizioni future: il token in posizione `t` puo' guardare solo da `0` a `t`.

La versione **multi-head** esegue l'attenzione in piu' sottospazi paralleli. Le teste vengono concatenate e proiettate di nuovo in `d_model`.

### MLP feed-forward

Dopo l'attenzione, ogni posizione passa in modo indipendente attraverso una piccola rete neurale:

```text
MLP(x) = W2 * activation(W1 * x + b1) + b2
```

L'MLP trasforma e combina le caratteristiche estratte dall'attenzione. Attivazioni comuni sono GELU o SiLU.

### Layer normalization e residui

Le connessioni residue permettono ai gradienti e alle informazioni di attraversare molti blocchi. La LayerNorm stabilizza la scala delle attivazioni. Nei LLM moderni e' comune la variante *pre-norm*, in cui la normalizzazione precede ogni sottolivello.

## Output: logits e probabilita'

Dopo l'ultimo blocco, una normalizzazione e una proiezione lineare producono un vettore di dimensione `vocab_size` per ogni posizione. Questi valori, non ancora normalizzati, sono i **logits**.

```text
hidden state -> linear head -> logits -> softmax -> P(prossimo token)
```

Prosegui con [l'addestramento](04-addestramento.md).
