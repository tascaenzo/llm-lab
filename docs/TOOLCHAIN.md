# Toolchain macOS e Linux

## Componenti scelti

| Componente | Scelta | Motivo |
|---|---|---|
| Linguaggio | C23 | Standard C moderno, con funzionalita' contemporanee e supporto nei compilatori aggiornati. |
| Build system | CMake >= 3.24 | Genera build native riproducibili. |
| Builder | Ninja | Veloce e identico su macOS e Linux. |
| Compilatore | Clang o GCC | Il progetto verifica queste due famiglie. |
| Formattazione | clang-format | Stile riproducibile. |
| Test | CTest + unittest | Test C e Python senza framework esterni. |
| CI | GitHub Actions | Compilazione e test su macOS e Linux. |

Il codice richiede C23, non estensioni specifiche di un compilatore. Servono
quindi versioni aggiornate di Clang o GCC; la CI rileva regressioni di
compatibilita' sulle piattaforme supportate.

## Installazione

### macOS

```sh
xcode-select --install
brew install cmake ninja clang-format
```

### Ubuntu / Debian

```sh
sudo apt update
sudo apt install build-essential cmake ninja-build clang-format
```

### Fedora

```sh
sudo dnf install gcc cmake ninja-build clang-tools-extra
```

### Windows

Windows non e' supportato in questa fase e non viene eseguito nella CI. CMake
rifiuta intenzionalmente la configurazione invece di produrre una build non
verificata. Il supporto verra' rivalutato soltanto quando potra' essere mantenuto
e testato in modo continuativo.

## Build

I preset CMake sono l'interfaccia comune tra sistemi operativi:

```sh
cmake --preset debug
cmake --build --preset debug
```

Per una build ottimizzata usare `release` al posto di `debug`.

## Verifiche locali

```sh
cmake --build --preset debug --target clean
cmake --build --preset debug
ctest --preset debug
make check-format
python3 -m compileall -q utils tests
```

Su macOS e Linux `make test` configura, compila ed esegue la stessa suite Debug.
Le fixture sono intenzionalmente piccole: i test non leggono il corpus Wikipedia
completo e non richiedono rete.

`compile_commands.json` e' generato nella directory della build; gli editor che lo supportano possono usarlo per analisi e autocomplete.

## Aggiungere dipendenze in futuro

Per ora la base non ha librerie esterne: anche il tokenizer BPE e' implementato nel repository. Se una dipendenza diventera' necessaria, verra' registrata con versione bloccata e istruzioni riproducibili; non aggiungeremo download impliciti nella fase di configurazione CMake.

## Preparazione del corpus multi-sorgente

Gli strumenti in `utils/corpus/` usano la libreria standard, con una sola
eccezione: `normalize_source.py` legge Parquet e richiede `pyarrow`.

```sh
python3 -m venv .venv
.venv/bin/pip install pyarrow
.venv/bin/python utils/corpus/normalize_source.py --help
```

La dipendenza vive solo nella preparazione dati offline. Il runtime di training
resta senza dipendenze: la build C non la vede e non la richiede.
