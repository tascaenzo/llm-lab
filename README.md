# llm-lab

Un laboratorio in C per capire e costruire, passo dopo passo, un piccolo language model autoregressivo.

Il progetto privilegia chiarezza e portabilita' tra macOS, Linux e Windows. Usa C23, lo standard C piu' recente, e non usa framework di deep learning: le strutture numeriche, il tokenizer e il Transformer verranno implementati solo quando serviranno.

La prima specifica implementativa e' [docs/tokenizer.md](docs/tokenizer.md).
La preparazione del corpus italiano e' descritta in [docs/corpus.md](docs/corpus.md).

## Stato iniziale

La toolchain e' pronta. Il primo modulo implementato e' un tokenizer Byte-level BPE: addestra merge da un corpus, salva un modello `.llmtok` e lo ricarica.

## Requisiti

- CMake 3.24 o superiore;
- Ninja;
- compilatore C con supporto C23: Clang, GCC o MSVC recente;
- Git (solo per clonare il progetto).
- Python 3.8 o superiore (solo per scaricare e preparare il corpus).

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
./build/debug/tokenizer_experiment
```

Dal menu puoi addestrare e salvare un vocabolario, codificare testo in ID o decodificare una lista di ID usando un file `.llmtok` esistente.

## Struttura

```text
apps/      eseguibili del progetto (oggi: llm-lab)
artifacts/ tokenizer addestrati e versionati
include/   header pubblici dei moduli implementati
src/       implementazione dei moduli implementati
utils/     piccole utility riproducibili per dati e sviluppo
data/      corpus locali: originali, puliti e derivati (non versionati)
cmake/     moduli della toolchain
docs/      istruzioni tecniche
wiki/      teoria e roadmap dell'LLM
```

## Qualita' e riproducibilita'

- I preset CMake definiscono build Debug e Release in modo identico sui tre sistemi.
- Gli avvisi importanti del compilatore sono abilitati per Clang, GCC e MSVC.
- `clang-format` impone uno stile consistente.

Consulta anche la [roadmap dell'LLM](wiki/07-roadmap-llm-da-zero.md).
