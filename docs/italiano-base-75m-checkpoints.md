# Italiano Base 75M — checkpoint pubblici

La release [Italiano Base 75M v1.0.0](https://github.com/tascaenzo/llm-lab/releases/tag/italiano-base-75m-v1.0.0)
raccoglie gli stati principali del pretraining del modello Italiano Base 75M.

Il tokenizer da usare con ogni checkpoint e'
`artifacts/tokenizers/italiano-v3.llmtok`, gia' versionato nel repository.

## Checkpoint disponibili

| Step | File | Uso |
|---:|---|---|
| 34.600 | [italiano-base-75m-step-034600.llmckpt](https://github.com/tascaenzo/llm-lab/releases/download/italiano-base-75m-v1.0.0/italiano-base-75m-step-034600.llmckpt) | Milestone iniziale per osservare la qualita' nelle prime fasi. |
| 110.494 | [italiano-base-75m-step-110494.llmckpt](https://github.com/tascaenzo/llm-lab/releases/download/italiano-base-75m-v1.0.0/italiano-base-75m-step-110494.llmckpt) | Checkpoint intermedio del training locale. |
| 327.000 | [latest.llmckpt](https://github.com/tascaenzo/llm-lab/releases/download/italiano-base-75m-v1.0.0/latest.llmckpt) | Milestone intermedia CUDA. |
| 483.000 | [cuda-resume-327k.llmckpt](https://github.com/tascaenzo/llm-lab/releases/download/italiano-base-75m-v1.0.0/cuda-resume-327k.llmckpt) | Checkpoint recuperato dalla sessione CUDA. |
| 624.362 | [cuda-final.llmckpt](https://github.com/tascaenzo/llm-lab/releases/download/italiano-base-75m-v1.0.0/cuda-final.llmckpt) | Stato dell'ultimo update del pretraining. |
| 624.362, best | [cuda-final-best.llmckpt](https://github.com/tascaenzo/llm-lab/releases/download/italiano-base-75m-v1.0.0/cuda-final-best.llmckpt) | **Checkpoint consigliato**: validation loss migliore, `2,545995`. |

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
  cuda-final-best.llmckpt \
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
