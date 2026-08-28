# Serving e inferenza incrementale

**Stato:** implementato su CPU, Metal e CUDA. CPU e il percorso host Metal sono
coperti dalla build locale; il kernel CUDA deve essere validato nella CI o su
una macchina NVIDIA prima del rilascio di quel backend.

Il percorso di chat non esegue piu' un forward completo su tutte le 512
posizioni a ogni token. `lm_decode_session` conserva per ogni layer le chiavi e
i valori gia' calcolati e processa soltanto la nuova posizione.

## Contratto

- una sessione appartiene a un modello, che deve restare vivo piu' a lungo;
- la capacita' massima non supera `context_length`;
- `prefill` azzera logicamente la cache e consuma il prompt;
- durante il prefill l'output head viene eseguito soltanto sull'ultimo token;
- `decode` aggiunge un token e produce logits FP32 `[1, vocabulary_size]`;
- quando la finestra e' piena la CLI ricostruisce la cache usando gli ultimi
  `context_length` token, preservando la semantica sliding-window precedente;
- un errore di esecuzione rende la sessione inutilizzabile fino a `reset`.

Ogni token acceleratore viene racchiuso in un solo batch backend. Le operazioni
intermedie rimangono sul device e il processo host legge soltanto una riga di
logits, non `context_length` righe.

Il loader `lm_model_load_checkpoint_for_inference` alloca soltanto i valori dei
parametri. Legge e verifica comunque i momenti AdamW presenti nel checkpoint per
preservarne checksum e controllo dei valori finiti, ma non li trasferisce al
device e non alloca gradienti o workspace di backward. Il loader di training
rimane separato e invariato.

## Uso

```sh
./build/release/llm-lab model chat \
  artifacts/models/italiano-chat-75m/best.llmckpt \
  artifacts/tokenizers/italiano-v3.llmtok 128 \
  "Spiegami la fotosintesi in modo semplice." \
  --backend cpu --temperature 0.7 --top-k 40 --seed 1
```

Standard error riporta separatamente token/s di prefill e decode. Il benchmark
ripetibile esegue warm-up, controlla che l'output deterministico non cambi e
produce JSON:

```sh
python3 utils/benchmarks/benchmark_generation.py \
  ./build/release/llm-lab \
  artifacts/models/italiano-base-75m/checkpoints/italiano-base-75m-v1-step-610000-validation-best.llmckpt \
  artifacts/tokenizers/italiano-v3.llmtok \
  --backend cpu --tokens 64 --warmup 1 --runs 5 \
  --output artifacts/benchmarks/generation-cpu.json
```

Su un checkpoint Italiano-Base-75M locale, prompt `Ciao`, seed 1 e quattro
token, il confronto manuale sullo stesso host e build Release ha misurato circa
`0,35 token/s` col vecchio forward completo e `53,5 token/s` col decoder cached.
Il numero e' una misura locale, non una promessa cross-hardware: per regressioni
si usa lo script sulla macchina target.

## Precisione e gate qualitativi

Il riferimento resta FP32. Il contratto runtime v1 non dichiara ancora storage
F16/BF16 o pesi INT8, quindi la CLI non simula una modalita' ridotta convertendo
silenziosamente i dati. Una futura precisione deve essere esplicita nel formato
checkpoint e nel backend e superare almeno:

1. parita' greedy token-per-token su una suite congelata;
2. errore massimo e medio dei logits rispetto a FP32, con soglie versionate;
3. nessuna regressione nei gate qualitativi della chat;
4. riduzione misurata di memoria o latenza sulla macchina target;
5. fallback FP32 sempre disponibile.

Quantizzare prima di questi gate renderebbe il modello piu' piccolo, non
necessariamente piu' veloce o migliore.
