# Modello Minimal: rete reale e banco di prova del runtime

## Stato attuale

Il **Modello Minimal** e' la prima rete addestrabile completa di llm-lab.
Riceve token reali del dataset, calcola logits e cross-entropy, esegue backward,
aggiorna i pesi con AdamW e supporta checkpoint, ripresa, valutazione e
generazione greedy.

La rete e' intenzionalmente piccola: un blocco causale, una sola head e nessun
MLP. Ha gia' dimostrato che la loss e la validation loss migliorano su testo
italiano. Non e' ancora un chatbot o un generatore di articoli; e' il banco di
prova che dimostra che dataset, modello, trainer e runtime funzionano insieme.

## Architettura

~~~text
input IDs [B,T]
  -> token embedding [V,C]
  -> RMSNorm [B,T,C]
  -> Q/K/V + RoPE [B,T,1,C]
  -> causal attention [B,T,1,C]
  -> output projection + residual [B,T,C]
  -> output head [B*T,V]
  -> cross-entropy con target [B*T]
~~~

Il backward percorre la sequenza inversa e propaga i gradienti fino
all'embedding. Ogni peso ha valore, gradiente e i due momenti AdamW.

Il flag --layers 0 esiste soltanto come baseline diagnostico senza attention:
non identifica una fase del progetto né una seconda famiglia di modelli.

## Perche' la prima rete e' piccola

La prima rete deve rendere visibili gli errori, non nasconderli in una grande
architettura. Usa pero' gli stessi concetti che resteranno nel modello grande:

- dataset autoregressivo .llmdat e batcher riproducibile;
- parametri F32, gradienti, cross-entropy e AdamW;
- checkpoint e ripresa;
- API runtime indipendente dall'hardware;
- configurazione che descrive profondita', head, hidden size, MLP e contesto.

L'obiettivo raggiunto e' quindi piu' importante di un esempio fittizio:
dimostra un training step reale e misurabile.

## CPU come riferimento, Metal per il training

~~~text
Modello Minimal / trainer
            |
            v
       runtime API llm_*
            |
            +-- CPU: riferimento numerico completo
            `-- Metal: training del Modello Minimal
~~~

CPU e' il riferimento numerico; Metal esegue l'intero training step del Modello
Minimal, incluse RMSNorm, RoPE, causal attention, backward e AdamW. Su stessi
input e seed il progetto continua a confrontare:

1. logits;
2. loss;
3. gradienti;
4. pesi dopo l'update AdamW.

I risultati Metal devono restare compatibili con CPU entro una tolleranza
dichiarata. Il checkpoint Metal a due milioni di step dimostra il percorso
end-to-end; la prossima applicazione di questi test e' il decoder scalabile.

## Crescita della stessa rete

La configurazione contiene gia' gli assi che fanno crescere il decoder:

| Campo | Stato Minimal | Evoluzione successiva |
|---|---:|---|
| hidden_size | configurabile | rappresentazioni piu' ampie |
| layer_count | configurabile | pila di blocchi |
| head_count | configurabile | multi-head attention |
| feed_forward_size | configurabile | MLP SwiGLU |
| context_length | configurabile | input piu' lunghi |

Il forward/backward e' generalizzato a piu' layer, multi-head e SwiGLU.
Trainer, checkpoint, dataset e confini CPU/Metal restano gli stessi: il
modello cresce per configurazione, non per sostituzione.

I costi principali sono:

- piu' layer: crescita approssimativamente lineare;
- hidden size maggiore: molte matrici crescono in modo quadratico;
- contesto maggiore: l'attention cresce circa con il quadrato del contesto.

Il target iniziale piu' grande e' [Italiano-Base-75M](14-italiano-base-75m.md):
il benchmark Metal determina il micro-batch sostenibile, non il numero nominale
di parametri.

## Criteri di completamento del decoder scalabile

Prima del training prolungato Base-75M, devono essere veri tutti questi punti:

- nessun fallback CPU durante lo step Metal;
- test di parita' CPU/Metal per forward, backward e update;
- checkpoint compatibile tra i backend;
- benchmark riproducibile con forme di training reali;
- nessuna regressione dei test CPU e del training sul dataset minuscolo.

La specifica operativa del target e' in
[Italiano-Base-75M](../docs/italiano-base-75m.md).
