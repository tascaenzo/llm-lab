# M1 — primo blocco Transformer causale

**Stato:** primo blocco causale CPU con RoPE, backward, checkpoint, ripresa,
generazione e valutazione implementati. MLP e scalabilita' dei layer seguono.

M1 conserva embedding, head lessicale, trainer, checkpoint e CLI di M0. Con
`layer_count=0` il comportamento resta M0; M1 richiede esattamente
`layer_count=1`, `head_count=1` e `feed_forward_size=0`.

## Forward

```text
hidden = embedding(input_ids)                    [B,T,C]
norm   = RMSNorm(hidden, attention_norm_weight)  [B,T,C]
Q      = norm @ query_weight                     [B,T,1,C]
K      = norm @ key_weight                       [B,T,1,C]
V      = norm @ value_weight                     [B,T,1,C]
Q,K    = RoPE(Q), RoPE(K)                         [B,T,1,C]
attn   = causal_attention(Q, K, V, 1/sqrt(C))    [B,T,1,C]
proj   = attn @ attention_output_weight           [B,T,C]
hidden = hidden + proj                            [B,T,C]
logits = reshape(hidden, [B*T,C]) @ output_weight [B*T,V]
```

RoPE usa tabelle deterministiche `[T,C/2]`, derivate dalla configurazione e
non salvate come parametri. Non aggiungiamo ancora MLP/SwiGLU: il blocco deve
isolare la causalita' e il backward dell'attention prima di aumentare il numero
di parametri.

`model train` usa M1 di default; `--layers 0` seleziona esplicitamente il
baseline M0. Checkpoint M0 e M1 restano distinguibili da `layer_count`.

## Configurazione attuale e scalabilita'

`lm_model_config` contiene fin dall'inizio i campi che descrivono un decoder
Transformer scalabile:

| Campo | Effetto sul modello futuro |
|---|---|
| `hidden_size` | ampiezza delle rappresentazioni e delle matrici del layer |
| `layer_count` | numero di blocchi Transformer in pila |
| `head_count` | teste di attention per blocco |
| `feed_forward_size` | dimensione intermedia dell'MLP SwiGLU |
| `context_length` | token elaborati nello stesso esempio |

Questi campi **non sono ancora interamente liberi**: M1 accetta solo
`layer_count=1`, `head_count=1`, `feed_forward_size=0` (oppure M0 con tutti a
zero). Il registry dinamico dei parametri e il checkpoint non dipendono pero'
da un array di pesi hard-coded: sono la base che permettera' a M2 di registrare
i parametri di ogni blocco senza cambiare il formato concettuale del trainer.

M2 aggiungera' per ogni layer una normalizzazione per attention e una per MLP,
proiezioni multi-head Q/K/V/output, SwiGLU e residual. Sara' quindi lo stesso
decoder-only language model a dimensioni diverse, non un modello parallelo o
una riscrittura separata.

Il costo non cresce in modo uniforme: aumentare i layer e' approssimativamente
lineare; allargare `hidden_size` aumenta molte matrici in modo quadratico; la
causal attention cresce approssimativamente con il quadrato di
`context_length`. Per questo M1 resta il riferimento piccolo per la parita'
CPU/Metal prima dei run lunghi M2.

## Parametri aggiuntivi

| Nome | Shape |
|---|---|
| `layers.0.attention_norm_weight` | `[C]` |
| `layers.0.query_weight` | `[C,C]` |
| `layers.0.key_weight` | `[C,C]` |
| `layers.0.value_weight` | `[C,C]` |
| `layers.0.attention_output_weight` | `[C,C]` |

Ogni parametro possiede gradiente, primo e secondo momento AdamW. Il registry
diventa dinamico: checkpoint e optimizer iterano il registry, non un array
hard-coded.

## Backward

Il gradiente della head torna nel residual; viene accumulato nel ramo identity
e nel ramo attention. Il backward segue l'ordine inverso:

```text
output projection -> attention backward -> Q/K/V linear -> RMSNorm backward
-> residual accumulate -> embedding scatter-add
```

## Gate

- gradient check delle proiezioni Q/K/V/output e RMSNorm;
- modificare un token futuro non modifica logits precedenti;
- overfit riproducibile di fixture minuscola;
- checkpoint M1 riprende con stessa sequenza di loss;
- M0 e M1 sono identificati dal checkpoint e un loader non interpreta mai
  parametri di una configurazione come quelli dell'altra.
