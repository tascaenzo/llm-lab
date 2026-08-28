# Corpus chat italiano v2 curato

Questo e' il corpus da usare per il prossimo SFT di `Italiano-Chat-75M`.
Non modifica il sorgente `data/italiano-chat-corpus-v1.jsonl`: e' una
selezione riproducibile e tracciabile, costruita per correggere il run chat v1.

## Artefatti

```text
data/derived/italiano-chat-corpus-v2/
  curation-report.json
  italiano-chat-corpus-v2.curated.jsonl
  italiano-chat-corpus-v2.context512.jsonl
  italiano-chat-corpus-v2.context512.{train,validation,test}.llmsft
```

Il corpus curato ha 1.777 istruzioni; gli split nativi contengono 1.593
esempi train, 89 validation e 95 test. Gli ID, la provenienza, la licenza e
lo stato di approvazione della riga d'origine restano in ogni record.

## Criteri applicati

- esclude integralmente i 500 esempi di
  `local-synthetic:chatgpt-generated-v2`: erano costruiti con un template e
  hanno introdotto formule ripetitive nel precedente checkpoint;
- conserva esclusivamente il primo scambio `user`/`assistant` completo di
  ogni conversazione: per `ita_conversations_v3` questo evita di addestrare
  su follow-up narrativi o incoerenti;
- richiede una domanda esplicita, lunghezze minime/massime, assenza di URL,
  volgarita', istruzioni non sicure e duplicati esatti;
- esclude per ora i domini medico, legale, politico e finanziario: per
  aggiungerli serve un lotto specifico, rivisto e con risposte prudenti;
- rende disponibile il system prompt standard in ogni record sorgente. La
  derivazione a 512 token ne mantiene un campione deterministico del 20%, in
  modo coerente con `model chat --system`.

Il resoconto completo, inclusi conteggi e rifiuti per motivo, e' in
`curation-report.json`.

## Rigenerazione

```sh
python3 utils/sft/build_curated_italiano_chat_v2.py \
  data/italiano-chat-corpus-v1.jsonl \
  data/derived/italiano-chat-corpus-v2/italiano-chat-corpus-v2.curated.jsonl \
  --report data/derived/italiano-chat-corpus-v2/curation-report.json

python3 utils/sft/derive_context512_chat.py \
  data/derived/italiano-chat-corpus-v2/italiano-chat-corpus-v2.curated.jsonl \
  data/derived/italiano-chat-corpus-v2/italiano-chat-corpus-v2.context512.jsonl \
  --max-user-chars 300 --max-assistant-chars 420 --max-system-chars 60

./build/release/llm-lab dataset sft-prepare \
  artifacts/tokenizers/italiano-v3.llmtok \
  data/derived/italiano-chat-corpus-v2/italiano-chat-corpus-v2.context512.jsonl \
  data/derived/italiano-chat-corpus-v2/italiano-chat-corpus-v2.context512 \
  --context 512
```

I comandi rifiutano di sovrascrivere i JSONL e il preparatore nativo verifica
che ogni conversazione tokenizzata rientri nel contesto.

## Limite noto

Questa e' una selezione automatica conservativa, non una certificazione della
veridicita' di ogni risposta. Prima di allargare il corpus, i nuovi lotti vanno
campionati e valutati contro prompt di test tenuti fuori dal training.
