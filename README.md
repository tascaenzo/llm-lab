# llm-lab

Un laboratorio in C per capire e costruire, passo dopo passo, un piccolo language model autoregressivo.

Il progetto privilegia chiarezza e portabilita' tra macOS, Linux e Windows. Usa C23, lo standard C piu' recente, e non usa framework di deep learning: ogni componente viene implementato quando diventa necessario e resta osservabile dalla CLI.

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
Per studiare l'intero percorso e il ruolo di ogni file consulta la
[guida al flusso dati e agli artefatti](wiki/09-flusso-dati-e-artefatti.md).

## Stato

La toolchain, il corpus e il tokenizer Byte-level BPE sono pronti. Il modulo
dataset divide i documenti in training, validation e test, crea artefatti binari
`.llmdat` e fornisce batch input/target al futuro modello. Il runtime tensoriale
CPU di riferimento implementa tensori FP32/U32, memoria, operazioni elementwise,
riduzioni, matmul, gather/scatter, softmax e cross-entropy. Il backend Metal
esegue lo stesso contratto sulla GPU Apple con pool dei buffer, batch asincroni
espliciti, metriche, kernel paralleli e matmul tiled FP32/FP16/BF16 con accumulo
FP32. Il prossimo incremento costruira' backward e layer neurali sopra queste
API. I sorgenti specifici dell'hardware
restano separati sotto `src/runtime/backends/`, cosi' CPU, Metal e futuri backend
CUDA non entrano nel codice del modello.

La suite prestazionale unificata accetta backend, precisione, operazioni, forme
e liste di thread configurabili; mostra una tabella e puo' produrre JSONL. Uso,
tempi GPU ed esempi sono nella
[specifica del backend Metal](docs/backend-metal.md#11-benchmark-riproducibile).

```sh
cmake --preset release -DLLM_LAB_BUILD_BENCHMARKS=ON
cmake --build --preset release --target runtime_benchmark
./build/release/utils/benchmarks/runtime_benchmark \
  --backend all --operations matmul --precision f16 \
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
[specifica Metal](docs/backend-metal.md#suite-prestazionale-rappresentativa).

## Requisiti

- CMake 3.24 o superiore;
- Ninja;
- compilatore C con supporto C23: Clang, GCC o MSVC recente;
- Git (solo per clonare il progetto).
- Python 3.8 o superiore (per utility del corpus e test automatici).

Le istruzioni d'installazione per ogni sistema operativo sono in [docs/TOOLCHAIN.md](docs/TOOLCHAIN.md).

## Comandi principali

```sh
cmake --preset debug
cmake --build --preset debug
./build/debug/llm-lab        # macOS/Linux
build\debug\llm-lab.exe         # Windows PowerShell
```

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
  artifacts/tokenizers/italiano-wikipedia-v1.llmtok
```

Il modello binario viene caricato una sola volta. Dal menu puoi convertire testo in
ID oppure una lista di ID separati da spazi nel testo originale. Il training resta
nel comando non interattivo `llm-lab tokenizer train`.

Per misurare compressione, velocita' e round-trip su un campione deterministico:

```sh
./build/release/llm-lab tokenizer evaluate \
  artifacts/tokenizers/italiano-wikipedia-v1.llmtok \
  1048576 corpus/italiano.txt
```

Il risultato JSON include byte, token, byte per token, durata, throughput e verifica del round-trip.

## Preparare il dataset del language model

Il comando seguente assegna ogni documento a uno split stabile, lo tokenizza e
scrive tre stream binari:

```sh
./build/debug/llm-lab dataset prepare \
  artifacts/tokenizers/italiano-wikipedia-v1.llmtok \
  data/clean/italiano-wikipedia-v1/documents.jsonl \
  data/derived/italiano-wikipedia-v1/lm/italiano-wikipedia-v1
```

La directory che contiene il prefisso di output deve gia' esistere. Il report JSON
finale mostra documenti e token prodotti per ogni split; durante il lavoro standard
error mostra percentuale, throughput ed ETA. Formato, token `<EOD>` e batcher sono
specificati in [docs/dataset.md](docs/dataset.md).

## Struttura

```text
apps/      eseguibili del progetto (oggi: llm-lab)
artifacts/ tokenizer addestrati e versionati
utils/benchmarks/ strumenti per misure prestazionali e confronto delle regressioni
include/   header pubblici dei moduli implementati
src/       implementazione dei moduli implementati
utils/     piccole utility riproducibili per dati e sviluppo
tests/     test C, test Python, integrazione CLI e fixture minime
data/      corpus locali: originali, puliti e derivati (non versionati)
cmake/     moduli della toolchain
docs/      specifiche implementative e istruzioni tecniche
wiki/      teoria e roadmap dell'LLM
```

## Qualita' e riproducibilita'

- I preset CMake definiscono build Debug e Release in modo identico sui tre sistemi.
- Gli avvisi importanti del compilatore sono abilitati per Clang, GCC e MSVC.
- `clang-format` impone uno stile consistente.
- CTest esegue test unitari C, pulizia del corpus e flussi CLI senza dipendenze esterne.

Consulta anche la [roadmap dell'LLM](wiki/07-roadmap-llm-da-zero.md).
