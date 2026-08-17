# llm-lab

Un laboratorio in C per capire e costruire, passo dopo passo, un piccolo language model autoregressivo.

Il progetto privilegia chiarezza e un percorso di sviluppo mirato ad Apple
Silicon, mantenendo la build CPU disponibile anche su Linux. Usa C23, lo
standard C piu' recente, e non usa framework di deep learning: ogni componente
viene implementato quando diventa necessario e resta osservabile dalla CLI.

L'[indice della documentazione implementativa](docs/README.md) raccoglie le
specifiche tecniche; la [wiki](wiki/README.md) spiega prima la teoria con un
percorso di lettura guidato.

La prima specifica implementativa e' [docs/tokenizer.md](docs/tokenizer.md).
La preparazione del corpus italiano e' descritta in [docs/corpus.md](docs/corpus.md).
La conversione in dati autoregressivi e' descritta in [docs/dataset.md](docs/dataset.md).
Il runtime tensoriale e' introdotto nella
[wiki](wiki/10-runtime-tensoriale.md) e specificato in
[docs/runtime-tensoriale.md](docs/runtime-tensoriale.md).
Il backend CPU parallelo e' spiegato nella
[wiki](wiki/11-backend-cpu.md) e definito tecnicamente in
[docs/backend-cpu.md](docs/backend-cpu.md).
Il primo backend GPU Metal e' introdotto nella
[wiki](wiki/12-backend-metal.md) e specificato in
[docs/backend-metal.md](docs/backend-metal.md).
Il backend CUDA, usato per proseguire il training su GPU in cloud partendo da un
checkpoint prodotto sul Mac, e' specificato in
[docs/backend-cuda.md](docs/backend-cuda.md).
Per studiare l'intero percorso e il ruolo di ogni file consulta la
[guida al flusso dati e agli artefatti](wiki/09-flusso-dati-e-artefatti.md).

## Stato

La toolchain, il corpus e il tokenizer Byte-level BPE sono pronti. Il modulo
dataset divide i documenti in training, validation e test, crea artefatti binari
`.llmdat` e fornisce batch input/target al Modello Minimal. E' una rete reale,
piccola e backend-agnostic, eseguita su CPU e Metal: embedding, blocco causale,
output head, cross-entropy, backward esplicito e AdamW. Il runtime tensoriale
CPU di riferimento implementa tensori FP32/U32, memoria, operazioni elementwise,
riduzioni, matmul, gather/scatter, softmax, cross-entropy e le primitive F32 di
training: matmul trasposta, accumulo, SiLU, RMSNorm, RoPE, attention GQA causale
con backward e AdamW. Il backend Metal dispone di pool dei buffer, batch
asincroni espliciti, metriche e tutte le primitive F32 richieste dal decoder
scalabile multi-layer, multi-head e SwiGLU di
[Italiano-Base-75M](docs/italiano-base-75m.md). Il runtime v1 accetta soltanto F32/U32: F16/BF16 sono
riservati e cast o mixed precision non fanno parte del contratto corrente.
Le prossime ottimizzazioni Metal saranno guidate dalle forme e dai colli di
bottiglia del modello reale. I sorgenti specifici dell'hardware
restano separati sotto `src/runtime/backends/`, cosi' CPU e Metal non entrano
nel codice del modello.

La suite prestazionale unificata accetta backend, operazioni, forme e liste di
thread configurabili; copre tutti i 27 workload CPU, mostra una
tabella e puo' produrre JSONL confrontabile con una baseline. Uso,
tempi GPU ed esempi sono nella
[specifica del backend Metal](docs/backend-metal.md).

```sh
cmake --preset release -DLLM_LAB_BUILD_BENCHMARKS=ON
cmake --build --preset release --target runtime_benchmark
./build/release/utils/benchmarks/runtime_benchmark \
  --backend all --operations matmul \
  --threads 1,2,4,8,auto --rows 512 --inner 512 --columns 512
```

Per riconoscere automaticamente la macchina e misurare tutti i kernel
supportati con un solo comando:

```sh
cmake --build --preset release --target runtime_benchmark_report
./build/release/utils/benchmarks/runtime_benchmark_report
```

Sono disponibili anche `--quick` per un controllo breve e `--full` per misure
piu' lunghe e stabili.

La suite rappresentativa per validare le prestazioni dopo una modifica dura
indicativamente 40–70 secondi, mostra l'avanzamento di ogni test e termina con
un riepilogo immediato CPU/Metal. Si avvia con:

```sh
cmake --build --preset release --target runtime_performance_suite
```

Baseline e confronto automatico sono descritti nella
[specifica Metal](docs/backend-metal.md).

## Requisiti

- CMake 3.24 o superiore;
- Ninja;
- compilatore C con supporto C23: Clang o GCC recente;
- Git (solo per clonare il progetto).
- Python 3.8 o superiore (per utility del corpus e test automatici).

Le istruzioni d'installazione per ogni sistema operativo sono in [docs/TOOLCHAIN.md](docs/TOOLCHAIN.md).

## Comandi principali

```sh
cmake --preset debug
cmake --build --preset debug
./build/debug/llm-lab        # macOS/Linux
```

Windows non e' supportato in questa fase: la configurazione CMake termina con
un errore esplicito. Il supporto potra' essere riaperto quando esisteranno una
necessita' concreta e una CI dedicata.

Oppure, su macOS/Linux, sono disponibili le scorciatoie:

```sh
make run
make test
make check-format
```

## Addestrare un tokenizer

Dopo la build, il comando seguente addestra un modello BPE e lo salva in un file portabile:

```sh
./build/debug/llm-lab tokenizer train italiano.llmtok 32000 corpus/italiano.txt
```

`32000` e' il massimo iniziale del tokenizer italiano. Su un corpus troppo piccolo il trainer si ferma prima, quando non restano piu' coppie da fondere. La scelta e' spiegata in [docs/tokenizer.md](docs/tokenizer.md).

Per aprire il laboratorio interattivo:

```sh
./build/debug/tokenizer_experiment \
  artifacts/tokenizers/italiano-wikipedia-v2.llmtok
```

Il modello binario viene caricato una sola volta. Dal menu puoi convertire testo in
ID oppure una lista di ID separati da spazi nel testo originale. Il training resta
nel comando non interattivo `llm-lab tokenizer train`.

Per misurare compressione, velocita' e round-trip su un campione deterministico:

```sh
./build/release/llm-lab tokenizer evaluate \
  artifacts/tokenizers/italiano-wikipedia-v2.llmtok \
  1048576 corpus/italiano.txt
```

Il risultato JSON include byte, token, byte per token, durata, throughput e verifica del round-trip.

## Preparare il dataset del language model

Il comando seguente assegna ogni documento a uno split stabile, lo tokenizza e
scrive tre stream binari:

```sh
./build/debug/llm-lab dataset prepare \
  artifacts/tokenizers/italiano-wikipedia-v2.llmtok \
  data/clean/italiano-wikipedia-v1/documents.jsonl \
  data/derived/italiano-wikipedia-v1/lm/italiano-wikipedia-v2
```

La directory che contiene il prefisso di output deve gia' esistere. Il report JSON
finale mostra documenti e token prodotti per ogni split; durante il lavoro standard
error mostra percentuale, throughput ed ETA. Formato, token `<EOD>` e batcher sono
specificati in [docs/dataset.md](docs/dataset.md).

## Addestrare il Modello Minimal

Il Modello Minimal usa `embedding -> RMSNorm -> Q/K/V + RoPE -> causal attention -> residual
-> output head`.
E' una rete con contesto causale, addestrata su CPU ma indipendente dal backend.
L'opzione `--layers 0` e' disponibile soltanto come baseline diagnostico.

```sh
./build/debug/llm-lab model train \
  data/derived/italiano-wikipedia-v1/lm/italiano-wikipedia-v2.train.llmdat \
  100 --batch-size 2 --context 32 --hidden 64 \
  --learning-rate 0.001 --seed 1
```

Il comando stampa loss e configurazione in JSON. Il sampler `shuffled`
percorre blocchi non sovrapposti, quindi un'epoca corrisponde davvero a un
passaggio sui token del corpus. Per conservare il risultato,
aggiungi `--checkpoint artifacts/models/m1.llmckpt`; per continuare da quel
file usa `--resume artifacts/models/m1.llmckpt --checkpoint ...`. Il checkpoint
salva pesi, momenti AdamW, step, configurazione e stato del batcher, cosi' la
sequenza dei batch prosegue identica. Un run lungo si ferma con `Ctrl-C`:
il comando completa lo step in corso, salva il checkpoint e riporta
`"interrupted":true`, quindi il lavoro fatto non va perso. Il trainer include scheduler, clipping,
sampler riproducibile e checkpoint periodici; la valutazione su validation e'
disponibile con model evaluate. I contratti tecnici sono in
[Modello Minimal](docs/model-minimal.md).

Prima del primo aggiornamento il comando valida tutto il `.llmdat` (checksum e
intervallo degli ID). Sul dataset Wikipedia la lettura di circa 5,8 GiB e' quindi
normale; `Verifica dataset` ne mostra avanzamento e ETA, poi `Training modello`
mostra step, loss e velocita'.

Per provare un checkpoint con sampling riproducibile:

```sh
./build/debug/llm-lab model generate artifacts/models/m1-step-10000.llmckpt \
  artifacts/tokenizers/italiano-wikipedia-v2.llmtok 32 "La capitale d'Italia" \
  --temperature 0.8 --top-k 40 --repetition-penalty 1.1 --seed 1
```

I valori mostrati sono anche i default. `--temperature` controlla la variabilita',
`--top-k` limita la scelta ai token piu' probabili, `--repetition-penalty` (almeno
1.0) penalizza i token gia' presenti nella finestra corrente e `--seed` rende il
risultato riproducibile. L'output UTF-8 valido viene scritto direttamente; solo
byte isolati o caratteri di controllo vengono mostrati come escape.

Il Modello Minimal ha un solo blocco causale: questa prova verifica il percorso checkpoint →
token → logits → testo, ma un modello piccolo e addestrato per pochi step non
produce ancora articoli o dialoghi affidabili.

Per misurare invece il checkpoint sullo split non visto:

```sh
./build/debug/llm-lab model evaluate \
  data/derived/italiano-wikipedia-v1/lm/italiano-wikipedia-v2.validation.llmdat \
  artifacts/models/m1-step-10000.llmckpt 100 --batch-size 2 --seed 1
```

Il Modello Minimal e' volutamente un riferimento ristretto: al momento accetta
un layer e una head, senza MLP; `--layers 0` e' soltanto il baseline diagnostico.
La configurazione contiene
gia' `hidden_size`, `layer_count`, `head_count` e `feed_forward_size`, ma il
decoder realmente scalabile arrivera' dopo la parita' Metal con multi-head,
SwiGLU e una pila di blocchi. Prima si porta il Modello Minimal a parita' di
training su Metal, poi potra'
usare gli stessi checkpoint, trainer e confini backend-agnostic.

## Struttura

```text
apps/      eseguibili del progetto (oggi: llm-lab)
artifacts/ tokenizer addestrati e versionati
utils/benchmarks/ strumenti per misure prestazionali e confronto delle regressioni
include/   header pubblici dei moduli implementati
src/       implementazione dei moduli implementati
src/model/ implementazione del Modello Minimal, parametri, layer e trainer
utils/     piccole utility riproducibili per dati e sviluppo
tests/     test C, test Python, integrazione CLI e fixture minime
data/      corpus locali: originali, puliti e derivati (non versionati)
cmake/     moduli della toolchain
docs/      specifiche implementative e istruzioni tecniche
wiki/      teoria e roadmap dell'LLM
```

## Qualita' e riproducibilita'

- I preset CMake definiscono build Debug e Release su macOS e Linux.
- Gli avvisi importanti del compilatore sono abilitati per Clang e GCC.
- `clang-format` impone uno stile consistente.
- CTest esegue test unitari C, pulizia del corpus e flussi CLI senza dipendenze esterne.

Consulta anche la [roadmap dell'LLM](wiki/07-roadmap-llm-da-zero.md).
