# Test

La suite usa CTest come unico punto di ingresso e non richiede framework esterni.

```sh
cmake --preset debug
cmake --build --preset debug
ctest --preset debug
```

La struttura separa le responsabilita':

- `tokenizer/`: API pubblica, formato modello, pretokenizzazione e trainer C;
- `python/`: pulizia MediaWiki e metadati degli artefatti;
- `integration/`: flusso reale della CLI `train -> evaluate`;
- `fixtures/`: corpus e modelli minimi, deterministici e versionabili.

I test non dipendono dal corpus Wikipedia locale. Nuove regressioni vanno ridotte
alla fixture piu' piccola capace di riprodurle.
