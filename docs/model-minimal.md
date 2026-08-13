# Modello Minimal — prima rete addestrabile del progetto

**Stato:** implementato su CPU. Il Modello Minimal e' la prima rete completa
del progetto: usa dataset reale, forward, loss, backward, AdamW, checkpoint,
ripresa, generazione greedy e valutazione. Il suo scopo e' validare il runtime
end-to-end con forme e carichi reali prima di ottimizzarlo e portare il training
interamente su Metal.

Non rappresenta una famiglia separata di modelli da mantenere nel tempo. E' il
riferimento piccolo della stessa architettura decoder-only che crescera' in
profondita', numero di head e MLP dopo la parita' CPU/Metal.

## Architettura attuale

```text
input IDs [B,T]
  -> token embedding [V,C]
  -> RMSNorm [B,T,C]
  -> Q/K/V + RoPE [B,T,1,C]
  -> causal attention [B,T,1,C]
  -> output projection + residual [B,T,C]
  -> output head [B*T,V]
  -> cross-entropy con target [B*T]
```

Il modello esegue oggi un solo blocco causale e una sola head; non ha ancora
MLP/SwiGLU. `--layers 0` resta soltanto un baseline diagnostico senza attention
per confronti e compatibilita' dei checkpoint: non e' una fase o un prodotto
distinto.

Il backward attraversa output head, residual, proiezione attention, attention,
Q/K/V, RMSNorm e embedding. Tutti i parametri sono F32 e possiedono gradiente,
primo momento e secondo momento AdamW.

## Confini e ownership

Il modello non conosce CPU, Metal, file `.llmdat` o CLI: usa esclusivamente
l'API runtime `llm_*`. Il trainer collega batcher autoregressivo, loss e
optimizer. Questa separazione e' la ragione per cui lo stesso modello puo'
diventare il test di parita' Metal senza introdurre condizioni hardware nel
codice dei layer.

```text
model / trainer
       |
       v
runtime API llm_*
       |
       +-- CPU: riferimento numerico completo
       `-- Metal: backend da completare e confrontare
```

- `lm_model` possiede parametri e workspace delle attivazioni, non il backend.
- `lm_trainer` non possiede dataset o modello.
- Il chiamante possiede input IDs, logits e gradiente dei logits.

## Struttura del codice

```text
include/model/model.h                API pubblica di modello e trainer
src/model/model.c                    lifecycle, configurazione e orchestration
src/model/parameter.c                parametro, gradiente e momenti AdamW
src/model/layers/embedding.c         embedding forward/backward
src/model/layers/output_head.c       proiezione lessicale e gradienti
src/model/layers/transformer.c       RMSNorm, Q/K/V, RoPE e attention causale
src/model/trainer.c                  batch, loss e optimizer step
src/model/checkpoint.c               salvataggio/ripresa atomici
```

## Configurazione: stato attuale e crescita futura

`lm_model_config` descrive gli assi della rete che rimarranno validi:

| Campo | Stato attuale | Ruolo nella crescita |
|---|---:|---|
| `vocabulary_size` | derivato dal dataset | dimensione embedding/head |
| `context_length` | configurabile | token elaborati per esempio |
| `hidden_size` | configurabile | ampiezza delle rappresentazioni |
| `layer_count` | 1 | numero di blocchi in pila |
| `head_count` | 1 | teste di attention per blocco |
| `feed_forward_size` | 0 | dimensione dell'MLP SwiGLU futuro |
| `seed` | configurabile | inizializzazione riproducibile |

Oggi la validazione ammette il blocco singolo (`layer_count=1`,
`head_count=1`, `feed_forward_size=0`). Il registry dei parametri e il formato
checkpoint sono dinamici; la prossima evoluzione generalizzera' il forward e
backward a piu' layer e head, aggiungendo due normalizzazioni e SwiGLU per
blocco. Sara' quindi la stessa rete a dimensioni diverse, non una riscrittura.

Il costo cresce circa linearmente con i layer, quadraticamente con
`hidden_size` per molte matrici, e quadraticamente con `context_length` per
l'attention. Per questo il primo obiettivo grande va affrontato solo dopo avere
misurato il runtime Metal con il Modello Minimal.

## CLI CPU

```sh
./build/debug/llm-lab model train TRAIN.llmdat STEPS \
  --batch-size 2 --context 32 --hidden 64 --layers 1 \
  --learning-rate 0.001 --seed 1 \
  --checkpoint artifacts/models/minimal-step-1000.llmckpt
```

Prima del primo update il comando valida integralmente `.llmdat`: checksum,
struttura e intervallo degli ID. Su dataset grandi questo controllo legge tutto
il file, quindi `Verifica dataset` puo' richiedere alcuni minuti. `Training
modello` inizia soltanto dopo e mostra step, loss, throughput ed ETA.

Il checkpoint contiene configurazione, pesi, momenti AdamW, step e stato del
batcher. Per continuare in modo riproducibile:

```sh
./build/debug/llm-lab model train TRAIN.llmdat 1000 \
  --resume artifacts/models/minimal-step-1000.llmckpt \
  --checkpoint artifacts/models/minimal-step-2000.llmckpt
```

## Valutazione e generazione

```sh
./build/debug/llm-lab model evaluate VALIDATION.llmdat \
  artifacts/models/minimal-step-2000.llmckpt 100 --batch-size 2 --seed 1

./build/debug/llm-lab model generate \
  artifacts/models/minimal-step-2000.llmckpt TOKENIZER.llmtok 32 \
  "La capitale d'Italia" \
  --temperature 0.8 --top-k 40 --repetition-penalty 1.1 --seed 1
```

La loss e la perplexity su validation sono il criterio numerico per decidere
se un training migliora; la generazione serve come controllo qualitativo.
La generazione usa sampling top-k riproducibile e penalizza le ripetizioni nella
finestra di contesto; questi valori sono i default e possono essere regolati da
riga di comando. Il testo UTF-8 valido viene stampato normalmente.
Con le dimensioni e gli step iniziali il testo non e' ancora un articolo o un
dialogo affidabile.

## Checkpoint di riferimento

Il checkpoint Metal del Modello Minimal a 2.000.000 step e' versionato per
consentire test di generazione e valutazioni riproducibili senza rieseguire il
training:

```text
artifacts/models/minimal-model/minimal-model-metal-step-2000000.llmckpt
```

La sua configurazione e' `vocabulary_size=32001`, `hidden_size=64`, un layer,
una head, contesto 32 e batch 2. Il checksum SHA-256 e'
`ce0957ca5aeeeb6960efc95a721d8539840904917737ccbf8147a876f6b0d803`.

## Gate per l'ottimizzazione Metal

Il Modello Minimal e' il test di integrazione per il runtime. Prima di lanciare
training prolungati o aumentare i parametri, Metal deve eseguire lo stesso
training step senza fallback CPU: RMSNorm, RoPE, causal attention, backward e
AdamW. I test di parita' confronteranno CPU e Metal su stessi input e seed:
logits, loss, gradienti e pesi dopo l'update devono essere compatibili entro
la tolleranza dichiarata.

Solo dopo questo gate si procedera' a ottimizzare batch, contesto, memoria e
kernel, e a rendere configurabili pila di layer, multi-head e SwiGLU per un
modello piu' grande.
