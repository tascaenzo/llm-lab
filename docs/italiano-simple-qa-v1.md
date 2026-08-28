# Italiano Simple QA v1

Corpus sintetico controllato per verificare se `Italiano-Base-75M` puo'
diventare affidabile su domande semplici. Non e' ancora un corpus chat generale.

## Contenuto

Il corpus contiene 723 esempi relativi a 253 gruppi fattuali:

| Categoria | Esempi |
|---|---:|
| Geografia | 253 |
| Scienza | 135 |
| Matematica | 140 |
| Cultura umanistica | 60 |
| Informatica | 60 |
| Lingua italiana | 45 |
| Istruzioni pratiche | 30 |

Ogni risposta e' breve e completa: la lunghezza massima e' 118 caratteri. Non
sono presenti system prompt, attualita', consigli medici, legali o finanziari.

Le parafrasi di uno stesso fatto sono assegnate insieme allo stesso split. In
questo modo una domanda sulla capitale di un paese non finisce nel train mentre
una sua semplice riscrittura finisce in validation o test. Gli split nativi
contengono 654 esempi train, 28 validation e 41 test.

## Artefatti

```text
data/derived/italiano-simple-qa-v1/
  italiano-simple-qa-v1.source.jsonl
  italiano-simple-qa-v1.manual-eval.jsonl
  italiano-simple-qa-v1.report.json
  italiano-simple-qa-v1.{train,validation,test}.llmsft
```

`manual-eval.jsonl` contiene parafrasi escluse dal training e la relativa
risposta attesa. Serve a misurare se il modello generalizza oltre la formulazione
esatta vista nel corpus.

## Rigenerazione

```sh
python3 utils/sft/generate_italiano_simple_qa_v1.py \
  data/derived/italiano-simple-qa-v1

./build/release/llm-lab dataset sft-prepare \
  artifacts/tokenizers/italiano-v3.llmtok \
  data/derived/italiano-simple-qa-v1/italiano-simple-qa-v1.source.jsonl \
  data/derived/italiano-simple-qa-v1/italiano-simple-qa-v1 \
  --context 512
```

Gli script rifiutano di sovrascrivere gli artefatti esistenti.

## Training proposto

```sh
./build/release/llm-lab model sft \
  data/derived/italiano-simple-qa-v1/italiano-simple-qa-v1.train.llmsft 160 \
  --base artifacts/models/italiano-base-75m/checkpoints/italiano-base-75m-v1-step-610000-validation-best.llmckpt \
  --backend metal --batch-size 2 --gradient-accumulation 4 \
  --learning-rate 5e-5 --min-learning-rate 5e-6 \
  --warmup-steps 15 --total-steps 160 \
  --beta1 0.9 --beta2 0.95 --epsilon 1e-8 \
  --weight-decay 0.01 --gradient-clip 1.0 \
  --validation data/derived/italiano-simple-qa-v1/italiano-simple-qa-v1.validation.llmsft \
  --validation-every 20 --validation-batches 14 \
  --checkpoint artifacts/models/italiano-chat-75m/runs/italiano-simple-qa-75m-v1-base610000/latest.llmckpt \
  --checkpoint-every 20 \
  --best-checkpoint artifacts/models/italiano-chat-75m/runs/italiano-simple-qa-75m-v1-base610000/best.llmckpt \
  --log artifacts/models/italiano-chat-75m/runs/italiano-simple-qa-75m-v1-base610000/training.jsonl
```

In questo run va selezionato `best.llmckpt` sulla validation. In inferenza non
va passato `--system`, perche' il corpus non contiene messaggi system. Il test
binario resta sigillato fino alla scelta del checkpoint.

## Criterio di successo

Il modello deve prima rispondere correttamente alle domande fattuali semplici
e alle parafrasi di `manual-eval.jsonl`. Solo dopo questo gate il corpus andra'
esteso con dialoghi, richieste ambigue e risposte piu' lunghe.
