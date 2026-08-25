# Modello Minimal — prima rete addestrabile del progetto

**Stato:** implementato su CPU e Metal. Il Modello Minimal e' la prima rete
completa del progetto: usa dataset reale, forward, loss, backward, AdamW,
checkpoint, ripresa, generazione e valutazione. Il suo scopo e' validare il
runtime end-to-end con forme e carichi reali prima di far crescere il decoder.

Non rappresenta una famiglia separata di modelli da mantenere nel tempo. E' il
riferimento piccolo della stessa architettura decoder-only, ora scalabile in
profondita', numero di head e dimensione dell'MLP.

## Architettura del riferimento minimo

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

Questa configurazione esegue un solo blocco causale, una sola head e non usa
MLP/SwiGLU. Il decoder supporta anche pile multi-layer, attenzione multi-head e
MLP SwiGLU. `--layers 0` resta soltanto un baseline diagnostico senza attention
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
       `-- Metal: backend verificato contro il riferimento CPU
```

- `lm_model` possiede parametri e workspace delle attivazioni, non il backend.
- `lm_trainer` non possiede dataset o modello.
- Il chiamante possiede input IDs, logits e gradiente dei logits.
- `lm_model_backward` **accumula** nel gradiente di ogni parametro e non lo
  azzera: solo `lm_model_zero_grad` lo fa. Le operazioni runtime di backward
  invece sovrascrivono i propri output, quindi ogni layer passa per un workspace
  e chiude con `llm_accumulate`. L'invariante e' verificata da un test: due
  backward consecutivi senza zero_grad devono dare esattamente il doppio del
  gradiente di uno, per tutti i parametri.

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

## Configurazione

`lm_model_config` descrive gli assi della rete che rimarranno validi:

| Campo | Stato attuale | Ruolo nella crescita |
|---|---:|---|
| `vocabulary_size` | derivato dal dataset | dimensione embedding/head |
| `context_length` | configurabile | token elaborati per esempio |
| `hidden_size` | configurabile | ampiezza delle rappresentazioni |
| `layer_count` | configurabile | numero di blocchi in pila |
| `head_count` | configurabile | teste di attention per blocco |
| `feed_forward_size` | configurabile | dimensione dell'MLP SwiGLU |
| `seed` | configurabile | inizializzazione riproducibile |

Il decoder accetta una pila di layer, multi-head attention e SwiGLU. La
configurazione `layer_count=1`, `head_count=1`, `feed_forward_size=0` resta
supportata esclusivamente per caricare e confrontare il checkpoint del Modello
Minimal; una configurazione scalabile richiede `layer_count > 0`,
`head_count > 0`, `feed_forward_size > 0` e `hidden_size % head_count == 0`.
Il registry dei parametri e il formato checkpoint sono dinamici: e' la stessa
rete a dimensioni diverse, non una riscrittura.

Il costo cresce circa linearmente con i layer, quadraticamente con
`hidden_size` per molte matrici, e quadraticamente con `context_length` per
l'attention. Per questo il primo obiettivo grande va affrontato solo dopo avere
misurato il runtime Metal con il Modello Minimal.

## CLI CPU

```sh
./build/debug/llm-lab model train TRAIN.llmdat STEPS \
  --batch-size 2 --context 32 --hidden 64 --layers 1 --heads 1 --ffn 0 \
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
I comandi `train`, `generate` ed `evaluate` risolvono il backend con la stessa
precedenza: `--backend`, `LLM_LAB_BACKEND` esportato, `.env` nella directory
corrente, infine CPU. L'evaluation non e' quindi piu' vincolata alla CPU e usa
il batching nativo quando il backend scelto e' Metal o CUDA.
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

## Riferimento Metal per il decoder scalabile

Il Modello Minimal e' il test di integrazione per il runtime. Il trainer Metal
raggruppa forward, backward e AdamW in un batch sul device; il checkpoint a due
milioni di step dimostra l'esecuzione end-to-end. Il prossimo gate applica la
stessa disciplina al decoder scalabile: CPU e Metal devono concordare su input
e seed identici per logits, loss, gradienti e pesi dopo l'update, entro le
tolleranze dichiarate.

Il target approvato per questa evoluzione e'
[Italiano-Base-75M](italiano-base-75m.md); il Modello Minimal resta il suo
riferimento di correttezza e non viene sostituito.
