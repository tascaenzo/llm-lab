# SFT sanity facts v1

Questo e' un probe diagnostico, non un corpus chat da promuovere. Verifica che
il modello da 75M riesca ad associare 20 domande fattuali molto semplici alle
rispettive risposte dopo SFT.

## Artefatti

```text
data/derived/sft-sanity-facts-v1/
  sft-sanity-facts-v1.source.jsonl
  sft-sanity-facts-v1.manual-tests.jsonl
  sft-sanity-facts-v1.{train,validation,test}.llmsft
```

Gli ID sono scelti in modo deterministico affinche' il preparatore nativo
produca esattamente 20 record train, 4 validation e 3 test.

## Run diagnostico

```sh
./build/release/llm-lab model sft \
  data/derived/sft-sanity-facts-v1/sft-sanity-facts-v1.train.llmsft 120 \
  --base artifacts/models/italiano-base-75m/checkpoints/italiano-base-75m-v1-step-610000-validation-best.llmckpt \
  --backend metal --batch-size 2 --gradient-accumulation 4 \
  --learning-rate 1e-4 --min-learning-rate 1e-5 \
  --warmup-steps 10 --total-steps 120 \
  --beta1 0.9 --beta2 0.95 --epsilon 1e-8 \
  --weight-decay 0.01 --gradient-clip 1.0 \
  --validation data/derived/sft-sanity-facts-v1/sft-sanity-facts-v1.validation.llmsft \
  --validation-every 20 --validation-batches 2 \
  --checkpoint artifacts/models/italiano-chat-75m/runs/sft-sanity-facts-v1/latest.llmckpt \
  --checkpoint-every 20 \
  --best-checkpoint artifacts/models/italiano-chat-75m/runs/sft-sanity-facts-v1/best.llmckpt \
  --log artifacts/models/italiano-chat-75m/runs/sft-sanity-facts-v1/training.jsonl
```

Per questo esperimento va valutato `latest.llmckpt`: l'obiettivo e' verificare
che il modello abbia imparato le 20 coppie train, non selezionare il minimo
della validation su quattro domande diverse. Se non risponde correttamente a
una netta maggioranza delle 20 domande viste, non va costruito un corpus grande
finche' la causa non e' stata isolata.
