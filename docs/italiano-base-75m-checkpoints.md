# Italiano Base 75M — checkpoint pubblici

La release [Italiano Base 75M v1.0.0](https://github.com/tascaenzo/llm-lab/releases/tag/italiano-base-75m-v1.0.0)
raccoglie gli stati principali del pretraining del modello Italiano Base 75M.

Il tokenizer da usare con ogni checkpoint e'
`artifacts/tokenizers/italiano-v3.llmtok`, gia' versionato nel repository.

## Checkpoint disponibili

Nel workspace locale, i nomi canonici sono in
`artifacts/models/italiano-base-75m/checkpoints/`. Non usare i vecchi nomi
generici `best.llmckpt` e `latest.llmckpt`: rimandano a milestone diverse.

| Step | File | Uso |
|---:|---|---|
| 34.600 | `italiano-base-75m-v1-step-034600.llmckpt` | Milestone iniziale. |
| 110.494 | `italiano-base-75m-v1-step-110494.llmckpt` | Milestone intermedia locale. |
| 327.000 | `italiano-base-75m-v1-step-327000-validation-best.llmckpt` | Migliore validation della milestone CUDA. |
| 483.000 | `italiano-base-75m-v1-step-483000-resume.llmckpt` | Punto di ripresa del run finale. |
| 610.000 | `italiano-base-75m-v1-step-610000-validation-best.llmckpt` | **Checkpoint consigliato**: validation loss migliore, `2,545995`. |
| 624.362 | `italiano-base-75m-v1-step-624362-final.llmckpt` | Stato dell'ultimo update del pretraining. |

Ogni file pesa circa 858 MiB. Le impronte sono disponibili in
[italiano-base-75m-v1.0.0-SHA256SUMS.txt](https://github.com/tascaenzo/llm-lab/releases/download/italiano-base-75m-v1.0.0/italiano-base-75m-v1.0.0-SHA256SUMS.txt).

## Verifica e generazione

Dopo il download, verifica l'integrita' con:

```bash
shasum -a 256 -c italiano-base-75m-v1.0.0-SHA256SUMS.txt
```

Per una generazione locale su Metal con il checkpoint consigliato:

```bash
LLM_LAB_BACKEND=metal ./build/release/llm-lab model generate \
  artifacts/models/italiano-base-75m/checkpoints/italiano-base-75m-v1-step-610000-validation-best.llmckpt \
  artifacts/tokenizers/italiano-v3.llmtok \
  192 \
  "Roma e' la capitale d'Italia." \
  --temperature 0.7 \
  --top-k 40 \
  --repetition-penalty 1.1 \
  --seed 1
```

Italiano Base 75M e' un modello di pretraining autoregressivo: sa generare
testo italiano piu' coerente degli snapshot iniziali, ma non e' ancora un
modello istruito per domande e risposte affidabili. Il passaggio successivo e'
un fine-tuning supervisionato.
