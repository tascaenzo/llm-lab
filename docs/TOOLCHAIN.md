# Toolchain multipiattaforma

## Componenti scelti

| Componente | Scelta | Motivo |
|---|---|---|
| Linguaggio | C23 | Standard C moderno, con funzionalita' contemporanee e supporto nei compilatori aggiornati. |
| Build system | CMake >= 3.24 | Genera build native per Ninja, Make e Visual Studio. |
| Builder | Ninja | Veloce, semplice e identico su macOS, Linux e Windows. |
| Compilatore | Clang, GCC o MSVC | Il progetto verifica esplicitamente queste tre famiglie. |
| Formattazione | clang-format | Stile riproducibile. |
| Test | CTest + unittest | Test C e Python senza framework esterni. |
| CI | GitHub Actions | Compilazione su tre sistemi operativi. |

Il codice richiede C23, non estensioni specifiche di un compilatore. Servono quindi versioni aggiornate di Clang, GCC o MSVC; la CI rilevera' subito eventuali regressioni di compatibilita'.

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

Installare Visual Studio 2022 Build Tools con il workload **Desktop development with C++**, poi in PowerShell:

```powershell
winget install Kitware.CMake Ninja-build.Ninja LLVM.LLVM
```

Aprire una shell "Developer PowerShell for VS 2022" per rendere disponibile `cl.exe`.

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
