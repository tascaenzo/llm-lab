# llm-lab

Un laboratorio in C per capire e costruire, passo dopo passo, un piccolo language model autoregressivo.

Il progetto privilegia chiarezza e portabilita' tra macOS, Linux e Windows. Usa C23, lo standard C piu' recente, e non usa framework di deep learning: le strutture numeriche, il tokenizer e il Transformer verranno implementati solo quando serviranno.

La prima specifica implementativa e' [docs/tokenizer.md](docs/tokenizer.md).

## Stato iniziale

La toolchain e' pronta e il codice contiene soltanto una CLI minima. Il prossimo passo sara' aggiungere il modulo tokenizer Byte-level BPE per il corpus italiano.

## Requisiti

- CMake 3.24 o superiore;
- Ninja;
- compilatore C con supporto C23: Clang, GCC o MSVC recente;
- Git (solo per clonare il progetto).

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

## Struttura

```text
apps/      eseguibili del progetto (oggi: llm-lab)
cmake/     moduli della toolchain
docs/      istruzioni tecniche
wiki/      teoria e roadmap dell'LLM
```

## Qualita' e riproducibilita'

- I preset CMake definiscono build Debug e Release in modo identico sui tre sistemi.
- Gli avvisi importanti del compilatore sono abilitati per Clang, GCC e MSVC.
- `clang-format` impone uno stile consistente.

Consulta anche la [roadmap dell'LLM](wiki/07-roadmap-llm-da-zero.md).
