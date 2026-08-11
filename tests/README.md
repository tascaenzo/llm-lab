# Test

La suite usa CTest come unico punto di ingresso e non richiede framework esterni.

```sh
cmake --preset debug
cmake --build --preset debug
ctest --preset debug
```

La struttura separa le responsabilita':

- `tokenizer/`: API pubblica, formato modello, pretokenizzazione e trainer C;
- `runtime/`: tensori, backend CPU/Metal e parita' delle operazioni numeriche;
- `python/`: pulizia MediaWiki e metadati degli artefatti;
- `integration/`: flusso reale della CLI `train -> evaluate`;
- `fixtures/`: corpus e modelli minimi, deterministici e versionabili.

Lo smoke test `benchmark.runtime.report_smoke` esegue il report hardware rapido.
Su macOS misura anche Metal quando il runner espone realmente il device; sugli
altri sistemi verifica il percorso CPU e lo stub portabile.
`benchmark.runtime.performance_suite_smoke` controlla invece che l'orchestratore
della suite prestazionale, il formato JSONL e la selezione automatica dei backend
restino funzionanti senza imporre soglie temporali instabili alla CI.

I test non dipendono dal corpus Wikipedia locale. Nuove regressioni vanno ridotte
alla fixture piu' piccola capace di riprodurle.
