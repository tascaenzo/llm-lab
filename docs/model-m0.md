# Modello M0 — language model minimo addestrabile

**Stato:** baseline CPU implementata e mantenuta per confronto. M0 e' il primo modello reale del
progetto: riceve token, calcola logits, loss, gradienti e aggiornamenti AdamW.
Non contiene ancora blocchi Transformer.

## Confine

M0 implementa esclusivamente:

```text
input IDs [B,T]
  -> token_embedding [V,C]
  -> hidden [B,T,C]
  -> output_weight [C,V]
  -> logits [B*T,V]
```

Il backward calcola e accumula i gradienti di `token_embedding` e
`output_weight`. Il modello non conosce CPU, Metal, file `.llmdat` o CLI: usa
solo l'API runtime `llm_*`. Il trainer collega invece il batcher del dataset,
cross-entropy e AdamW.

## Struttura del codice

```text
include/model/model.h             API pubblica di modello e trainer
src/model/model.c                 lifecycle, configurazione e orchestration forward/backward
src/model/parameter.c             valore, gradiente e momenti AdamW di un parametro
src/model/layers/embedding.c      gather e scatter-add dell'embedding
src/model/layers/output_head.c    proiezione lineare e relativi gradienti
src/model/trainer.c               batch autoregressivo, loss e optimizer step
```

Un layer non alloca né seleziona un backend. `model.c` possiede workspace e
parametri; `trainer.c` possiede batcher, step e iperparametri. Questa divisione
permette di aggiungere RMSNorm, attention e SwiGLU come nuovi file sotto
`layers/`, senza inserire condizioni hardware nei layer esistenti.

## Configurazione e parametri

`lm_model_config` include gia' i campi futuri. In M0 devono valere:

```text
vocabulary_size > 0
context_length > 0
hidden_size > 0
layer_count = head_count = feed_forward_size = 0
```

I due parametri sono F32 con gradienti e due momenti F32 distinti:

| Nome | Shape |
|---|---|
| `token_embedding` | `[V,C]` |
| `output_weight` | `[C,V]` |

L'inizializzazione e il batcher ricevono seed espliciti. `lm_model_zero_grad`
deve essere chiamata prima del backward di un normale training step;
`lm_model_backward` accumula i gradienti, permettendo una futura estensione a
micro-batch.

## Ownership

- `lm_model` non possiede il backend; il backend deve vivere fino alla sua
  distruzione.
- Il modello possiede parametri e workspace delle attivazioni.
- Il chiamante possiede i tensori `input_ids`, logits e gradienti dei logits.
- `lm_trainer` non possiede dataset o modello; entrambi devono restare vivi
  fino alla distruzione del trainer.

## CLI CPU

```sh
./build/debug/llm-lab model train TRAIN.llmdat STEPS --layers 0 \
  --batch-size 2 --context 32 --hidden 64 \
  --learning-rate 0.001 --seed 1
```

Il comando accetta soltanto un artefatto `.train.llmdat`, valida integralmente
il file in apertura e stampa un report JSON con loss finale e configurazione.
Su dataset grandi questa verifica include checksum e tutti gli ID, quindi puo'
richiedere tempo: `Verifica dataset` mostra byte, velocita' ed ETA prima che
inizi il primo step. `Training modello` mostra poi step, loss, throughput ed
ETA.

Per mantenere il training, salva un checkpoint atomico finale:

```sh
./build/debug/llm-lab model train TRAIN.llmdat 1000 \
  --checkpoint artifacts/models/m0-step-1000.llmckpt
```

Il file contiene configurazione, pesi, momenti AdamW, step e stato del batcher.
Si riprende senza ripassare opzioni di modello o trainer:

```sh
./build/debug/llm-lab model train TRAIN.llmdat 1000 \
  --resume artifacts/models/m0-step-1000.llmckpt \
  --checkpoint artifacts/models/m0-step-2000.llmckpt
```

Un checkpoint sostituisce atomicamente l'eventuale file finale con lo stesso
nome. Validation, scheduler e clipping restano nella milestone di training
affidabile.

## Generazione M0

```sh
./build/debug/llm-lab model generate \
  artifacts/models/m0-step-2000.llmckpt \
  artifacts/tokenizers/italiano-wikipedia-v1.llmtok \
  32 "La capitale d'Italia"
```

Il comando carica direttamente il checkpoint, senza rileggere il dataset, e
stampa prompt più token generati greedy. Per mantenere l'output sempre sicuro
in terminale, i byte non ASCII stampabili appaiono come `\xNN`. Il tokenizer deve corrispondere al
vocabolario del checkpoint. M0 usa la stessa API di forward del futuro modello,
ma senza attention ogni nuovo token dipende soltanto dall'ultimo token: l'output
non va quindi interpretato come una conversazione coerente.

## Valutazione

```sh
./build/debug/llm-lab model evaluate \
  data/derived/italiano-wikipedia-v1/lm/italiano-wikipedia-v1.validation.llmdat \
  artifacts/models/m0-step-2000.llmckpt \
  100 --batch-size 2 --seed 1
```

Il report JSON contiene loss media e perplexity su batch deterministici dello
split validation. A parita' di checkpoint, batch size, numero di batch e seed,
questo e' il confronto numerico da usare prima di valutare esempi generati.

## Verifica

`tests/model/test_model.c` esegue gradient check a differenze finite per
embedding e output head, verifica il forward e la riproducibilita' del seed.
L'integrazione CLI costruisce un dataset minimo deterministico e richiede che
il training lo overfitti riducendo la loss.
