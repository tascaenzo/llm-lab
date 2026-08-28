# Corpus chat italiano v1

Il corpus chat attualmente in costruzione e' un unico file:

```text
data/italiano-chat-corpus-v1.jsonl
```

Contiene **6.623 conversazioni** e **22.114 risposte assistant**. Ogni riga e'
un oggetto JSON indipendente con `id`, `source`, `license`, `approval_status`
e `messages`.

## Regole per aggiungere un nuovo lotto

1. genera o raccogli il lotto separatamente;
2. verifica che ogni riga sia JSON valido, con una domanda `user` seguita da
   una risposta `assistant`;
3. elimina duplicati, prompt meta, quiz a scelta multipla, fatti inventati e
   risposte rischiose o dipendenti da dati aggiornati;
4. assegna un `id` unico e registra in `source` il modello docente o la fonte;
5. dopo il controllo manuale e l'approvazione, aggiungi le righe valide in
   `data/italiano-chat-corpus-v1.jsonl`.

Esempio di riga:

```json
{"id":"locale-tema-casa-0001","source":"local-synthetic:chatgpt","license":"USER_APPROVED","approval_status":"approved","messages":[{"role":"user","content":"Come posso organizzare una piccola dispensa?"},{"role":"assistant","content":"Inizia dividendo gli alimenti per categoria e controlla prima le scadenze. Metti quelli da usare prima davanti, usa contenitori etichettati e lascia uno spazio dedicato alle scorte."}]}
```

Il corpus corrente e' approvato per il training interno: tutte le righe hanno
`approval_status: "approved"`. Le 6.565 conversazioni sintetiche autorizzate
dall'utente sono marcate `USER_APPROVED`; le restanti 58 conservano la licenza
`Apache-2.0`. Gli ID aggiunti alle conversazioni ChatGPT sono deterministici e
seguono il formato `local-synthetic:chatgpt:<numero-riga>` o
`local-synthetic:chatgpt-v2:<numero>`.

Il lotto v2 aggiunge 500 dialoghi multi-turno (1.000 risposte assistant) e 100
messaggi `system`. E' rigenerabile con
`utils/sft/generate_italiano_chat_v2.py`.

Il checkpoint base ha un contesto di 512 token. Per il training, il primo
turno di ogni conversazione viene mantenuto come richiesta autonoma; i turni
successivi includono anche lo scambio `user`/`assistant` precedente. I testi
che non entrano nel contesto sono compattati esplicitamente e la derivazione e'
registrata in `provenance`. Il comando e' riproducibile:

```sh
python3 utils/sft/derive_context512_chat.py \
  data/italiano-chat-corpus-v1.jsonl \
  data/derived/italiano-chat-corpus-v1/italiano-chat-corpus-v1.context512-windowed.jsonl
```

Ogni nuovo lotto deve comunque riportare provenienza e licenza o approvazione
esplicita prima di entrare nel corpus.
