# Italiano Simple QA v2

Corpus SFT sintetico controllato per insegnare a `Italiano-Base-75M` a
rispondere in modo breve, corretto e pertinente a richieste italiane semplici.
La v2 amplia la copertura della v1 e introduce esplicitamente richieste
ambigue, entita' inventate e informazioni che richiedono una verifica
aggiornata.

## Contenuto

Il corpus contiene 5.551 esempi appartenenti a 584 gruppi semantici
indipendenti:

| Categoria | Esempi |
|---|---:|
| Matematica | 1.060 |
| Scienza | 705 |
| Geografia | 661 |
| Istruzioni pratiche | 620 |
| Incertezza e verifica | 500 |
| Informatica | 480 |
| Cultura umanistica | 430 |
| Conoscenze quotidiane | 300 |
| Scrittura e riscrittura | 290 |
| Richieste ambigue | 250 |
| Lingua italiana | 135 |
| Informazioni variabili | 120 |

I 5.551 esempi non rappresentano 5.551 fatti diversi: ogni gruppo contiene
piu' formulazioni controllate della stessa richiesta. Il numero da usare per
misurare la varieta' concettuale e' quindi 584.

Le risposte hanno al massimo 125 caratteri. Non sono presenti system prompt,
risposte troncate o segnaposto. Tutti gli esempi riportano licenza
`USER_APPROVED` e stato `approved`.

## Separazione dei dati

Tutte le parafrasi di uno stesso gruppo sono assegnate allo stesso split. Il
preparatore nativo produce:

| Split | Esempi | Token supervisionati |
|---|---:|---:|
| Train | 5.032 | 109.886 |
| Validation | 268 | 6.177 |
| Test | 251 | 5.278 |

Il file `manual-eval.jsonl` contiene inoltre 529 prompt esclusi dal corpus di
training. Il test binario e la valutazione manuale non vanno usati per scegliere
iperparametri o checkpoint.

## Artefatti

```text
data/derived/italiano-simple-qa-v2/
  italiano-simple-qa-v2.source.jsonl
  italiano-simple-qa-v2.manual-eval.jsonl
  italiano-simple-qa-v2.report.json
  italiano-simple-qa-v2.{train,validation,test}.llmsft
```

## Rigenerazione

Il generatore rifiuta di sovrascrivere file esistenti. Per una rigenerazione
intenzionale bisogna prima spostare o eliminare esplicitamente la directory di
output.

```sh
python3 utils/sft/generate_italiano_simple_qa_v2.py \
  data/derived/italiano-simple-qa-v2

./build/release/llm-lab dataset sft-prepare \
  artifacts/tokenizers/italiano-v3.llmtok \
  data/derived/italiano-simple-qa-v2/italiano-simple-qa-v2.source.jsonl \
  data/derived/italiano-simple-qa-v2/italiano-simple-qa-v2 \
  --context 512
```

## Training proposto

Il run parte dal checkpoint completo del modello base, non da un precedente
fine-tuning. Con batch effettivo 16, un'epoca corrisponde a circa 315 step; 600
step sono quindi un limite massimo di circa 1,9 epoche. Il modello da valutare
e' sempre `best.llmckpt`, non necessariamente `latest.llmckpt`.

```sh
mkdir -p artifacts/models/italiano-chat-75m/runs/italiano-simple-qa-75m-v2-base610000

./build/release/llm-lab model sft \
  data/derived/italiano-simple-qa-v2/italiano-simple-qa-v2.train.llmsft 600 \
  --base artifacts/models/italiano-base-75m/checkpoints/italiano-base-75m-v1-step-610000-validation-best.llmckpt \
  --backend metal --batch-size 2 --gradient-accumulation 8 \
  --learning-rate 3e-5 --min-learning-rate 3e-6 \
  --warmup-steps 50 --total-steps 600 \
  --beta1 0.9 --beta2 0.95 --epsilon 1e-8 \
  --weight-decay 0.01 --gradient-clip 1.0 \
  --validation data/derived/italiano-simple-qa-v2/italiano-simple-qa-v2.validation.llmsft \
  --validation-every 25 --validation-batches 50 \
  --checkpoint artifacts/models/italiano-chat-75m/runs/italiano-simple-qa-75m-v2-base610000/latest.llmckpt \
  --checkpoint-every 25 \
  --best-checkpoint artifacts/models/italiano-chat-75m/runs/italiano-simple-qa-75m-v2-base610000/best.llmckpt \
  --log artifacts/models/italiano-chat-75m/runs/italiano-simple-qa-75m-v2-base610000/training.jsonl
```

In inferenza non va usato `--system`, perche' il corpus non contiene messaggi
di sistema. Per una prima prova deterministica sono adatti temperatura `0.1`,
`top-k 5` e al massimo 60 token.

## Criterio di successo

La valutazione deve coprire separatamente almeno questi comportamenti:

- risposta corretta a conoscenze semplici non viste nella stessa formulazione;
- istruzione pratica pertinente e concisa;
- richiesta di dettagli quando la domanda e' ambigua;
- rifiuto di inventare una capitale o un'entita' inesistente;
- richiesta di verifica per ruoli, prezzi, meteo e versioni correnti;
- assenza di ripetizioni e deviazioni fuori tema.

Questo corpus non rende ancora il modello un assistente generale. Se supera il
gate, il passo successivo e' aggiungere dialoghi multi-turno e risposte piu'
articolate mantenendo gli stessi controlli di qualita'.
