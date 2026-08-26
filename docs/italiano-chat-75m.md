# Italiano-Chat-75M — consolidamento e supervised fine-tuning

**Stato:** la pipeline SFT e' implementata su CPU, Metal e CUDA. Comprende
formato conversazionale versionato, loss solo sui token assistant, accumulo
normalizzato per token supervisionato, validation, checkpoint riprendibili e
generazione con protocollo chat. Per ottenere un checkpoint utile restano da
fornire il corpus reale, eseguire i pilot e superare i criteri qualitativi.

Questa fase non aumenta layer o parametri. Parte da `Italiano-Base-75M` e ne
consolida comportamento e operativita'. Aggiungere capacita' architetturale
prima di misurare i limiti del modello attuale renderebbe piu' costoso capire
se gli errori dipendono da dati, training o dimensione.

## Cosa viene addestrato

Il tokenizer testuale non cambia. I sette ID gia' riservati dal modello base
sono interpretati dal protocollo chat v1, dato un vocabolario tokenizer `V`:

| ID | Significato |
|---:|---|
| `V` | fine documento del pretraining |
| `V + 1` | `<|system|>` |
| `V + 2` | `<|user|>` |
| `V + 3` | `<|assistant|>` |
| `V + 4` | `<|end|>` |
| `V + 5` | `<|pad|>` |
| `V + 6`, `V + 7` | riservati |

Una conversazione viene serializzata così:

```text
<|system|> istruzione di sistema <|end|>
<|user|> domanda <|end|>
<|assistant|> risposta <|end|>
```

La cross entropy usa una maschera `U32` per posizione. Hanno mask `1` soltanto
il contenuto assistant e il suo `<|end|>`; system, user, token di ruolo e
padding hanno mask `0`. Con gradient accumulation ogni micro-batch produce una
somma e l'update divide una volta per il numero totale di token supervisionati.
In questo modo esempi con risposte di diversa lunghezza hanno il peso corretto.

## Schema del corpus

Il file sorgente e' JSONL UTF-8, una conversazione per riga:

```json
{"id":"fonte-000001","source":"nome-dataset/versione","license":"CC-BY-4.0","messages":[{"role":"system","content":"Sei un assistente utile."},{"role":"user","content":"Perché il cielo è blu?"},{"role":"assistant","content":"La luce blu viene diffusa..."}]}
```

Contratti obbligatori:

- `id`, `source` e `license` sono stringhe non vuote; gli ID sono unici;
- `system` e' facoltativo e, se presente, compare soltanto all'inizio;
- dopo system i ruoli alternano `user`, `assistant` e terminano con assistant;
- i contenuti sono non vuoti;
- una conversazione deve entrare interamente in `context + 1` token: non viene
  troncata silenziosamente;
- lo split 90/5/5 dipende da `FNV-1a-64(id) mod 10000`, quindi e' stabile e
  impedisce che la stessa conversazione cambi split tra run.

Prima della tokenizzazione si verifica struttura, provenienza e distribuzione:

```sh
python3 utils/sft/validate_conversations.py \
  data/clean/italiano-sft-v1/conversations.jsonl \
  --report data/clean/italiano-sft-v1/validation-report.json
```

Il limite tokenizzato autorevole viene verificato dal comando nativo:

```sh
./build/release/llm-lab dataset sft-prepare \
  artifacts/tokenizers/italiano-v3.llmtok \
  data/clean/italiano-sft-v1/conversations.jsonl \
  data/derived/italiano-sft-v1/italiano-sft-v1 \
  --context 512
```

Produce tre file con checksum e dimensione fissa:
`*.train.llmsft`, `*.validation.llmsft`, `*.test.llmsft`. Ogni esempio contiene
input ID, target ID e mask. L'header registra checksum del tokenizer, protocollo,
contesto, conteggi di esempi e token supervisionati.

## Strategia dati

Per un 75M la qualita' e la coerenza contano piu' del volume indiscriminato.
Il primo corpus candidato dovrebbe combinare istruzioni generali, dialoghi
brevi, trasformazione del testo, ragionamento elementare, domande chiarificatrici
e rifiuti sicuri. Ogni sorgente deve avere una licenza verificabile; dati
personali, conversazioni private e output sintetici non revisionati non entrano
nel training.

Prima di un run lungo:

1. deduplicare semanticamente oltre al controllo degli ID;
2. revisionare manualmente campioni per sorgente e categoria;
3. misurare lunghezza e numero di token assistant per split;
4. rimuovere sovrapposizioni con la suite di valutazione;
5. congelare corpus, manifest e checksum.

Non esiste nel repository un corpus inventato automaticamente: la pipeline e'
pronta, ma la qualita' del prodotto dipende da dati realmente autorizzati e
revisionati.

## Pilot e training

Il primo pilot deve confermare che il checkpoint base, il tokenizer e il
dataset abbiano vocabolario e contesto identici. Per un run nuovo lo stato AdamW
del pretraining viene azzerato; `--resume` ripristina invece modello, optimizer,
scheduler e posizione del batcher SFT.

```sh
./build/release/llm-lab model sft \
  data/derived/italiano-sft-v1/italiano-sft-v1.train.llmsft 200 \
  --base artifacts/models/italiano-base-75m/best.llmckpt \
  --backend metal --batch-size 2 --gradient-accumulation 8 \
  --learning-rate 3e-5 --min-learning-rate 3e-6 \
  --warmup-steps 20 --total-steps 200 \
  --beta1 0.9 --beta2 0.95 --epsilon 1e-8 \
  --weight-decay 0.01 --gradient-clip 1.0 \
  --validation data/derived/italiano-sft-v1/italiano-sft-v1.validation.llmsft \
  --validation-every 25 --validation-batches 100 \
  --checkpoint artifacts/models/italiano-chat-75m/pilot-latest.llmckpt \
  --checkpoint-every 25 \
  --best-checkpoint artifacts/models/italiano-chat-75m/pilot-best.llmckpt \
  --log artifacts/models/italiano-chat-75m/pilot.jsonl
```

Il pilot serve a scegliere learning rate, batch effettivo e durata. Non si
seleziona il checkpoint sulla loss train: si usa il minimo validation e si
controlla la qualita' delle generazioni. Per riprendere esattamente:

```sh
./build/release/llm-lab model sft \
  data/derived/italiano-sft-v1/italiano-sft-v1.train.llmsft 200 \
  --resume artifacts/models/italiano-chat-75m/pilot-latest.llmckpt \
  --backend metal \
  --validation data/derived/italiano-sft-v1/italiano-sft-v1.validation.llmsft \
  --validation-every 25 --validation-batches 100 \
  --checkpoint artifacts/models/italiano-chat-75m/pilot-latest.llmckpt
```

## Inferenza e valutazione

La CLI applica lo stesso template del training, esclude dalla generazione tutti
i token di ruolo e padding, termina su `<|end|>` e usa il decoder incrementale
con KV cache descritto in [Serving e inferenza incrementale](serving-inference.md):

```sh
./build/release/llm-lab model chat \
  artifacts/models/italiano-chat-75m/best.llmckpt \
  artifacts/tokenizers/italiano-v3.llmtok 128 \
  "Spiegami la fotosintesi in modo semplice." \
  --backend metal --temperature 0.7 --top-k 40 \
  --repetition-penalty 1.1 --seed 1
```

La suite versionata si esegue con parametri fissi e produce un JSONL che include
checksum di modello, tokenizer e prompt:

```sh
python3 utils/sft/evaluate_chat.py \
  ./build/release/llm-lab \
  artifacts/models/italiano-chat-75m/best.llmckpt \
  artifacts/tokenizers/italiano-v3.llmtok \
  evaluations/italiano-chat-75m-prompts.jsonl \
  artifacts/models/italiano-chat-75m/evaluation.jsonl \
  --backend metal --tokens 128
```

Ogni risposta va valutata almeno per aderenza, italiano, formato, ripetizione,
correttezza evidente e sicurezza. La suite qualitativa non sostituisce la loss
assistant sul validation/test split.

## Gate prima di chiamarlo “funzionante”

Il checkpoint candidato deve soddisfare tutti i gate:

1. preparazione e training superano i test CPU e la parita' del backend usato;
2. la loss assistant validation migliora rispetto al checkpoint base e non
   diverge; il test resta sigillato fino alla selezione finale;
3. nessuna regressione grave di italiano o collasso ripetitivo sulla suite;
4. almeno il 90% dei test di formato semplici e' rispettato;
5. richieste ambigue producono domande chiarificatrici e le richieste dannose
   della suite non ricevono istruzioni operative;
6. checkpoint, tokenizer, dati, manifest, commit, comandi e metriche hanno
   checksum e sono riproducibili.

Il decode incrementale FP32 e' il riferimento di qualita'. Mixed precision e
quantizzazione restano opzioni separate: prima dell'adozione devono superare la
parita' di logits, la suite qualitativa e una misura hardware che dimostri un
vantaggio reale. Anche l'aumento di layer/parametri va deciso soltanto con error
analysis che mostri un limite di capacita', non un problema di dati o allineamento.
