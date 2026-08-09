# Corpus italiano — piano operativo

## Che cos'e' il corpus

Il corpus e' l'insieme dei testi da cui il tokenizer impara quali sequenze di byte
sono frequenti. Non e' un dizionario e non deve essere una raccolta casuale di
testi incollati insieme.

La qualita' del tokenizer dipende dal corpus: se il corpus contiene soprattutto
enciclopedia, il vocabolario sara' molto adatto a quell'italiano e meno a dialoghi,
codice o narrativa. Per questo registriamo sempre origine e trasformazioni dei
dati.

Questa fase serve per il tokenizer. Quando addestreremo un language model,
serviranno inoltre una selezione del corpus piu' ampia e split separati di training,
validazione e test.

## Corpus v1

Partiamo da due fonti con provenienza chiara:

1. **Wikipedia in italiano**: testo enciclopedico contemporaneo. Usiamo il dump
   ufficiale `pages-articles`, non lo scraping delle pagine web.
2. **Wikisource in italiano**: testi letterari e storici. Lo aggiungeremo dopo
   avere completato e verificato la pipeline su Wikipedia.

Wikipedia rende disponibile il dump aggiornato in un URL stabile. Al momento della
scrittura, il file completo compresso e' circa 4 GiB: non e' un download da avviare
per caso. Wikisource ammette testi di pubblico dominio o con licenza libera
compatibile con CC BY-SA; la licenza effettiva resta comunque un dato da registrare
per ogni sorgente del corpus.

Non includiamo inizialmente raccolte web aggregate: rendono meno semplice sapere
da quale sito provenga ogni testo e con quale licenza possa essere riutilizzato.

Riferimenti:

- <https://dumps.wikimedia.org/itwiki/latest/>
- <https://it.wikisource.org/wiki/Wikisource:Cos%27%C3%A8_Wikisource%3F>
- <https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use/Summary/en>

## Struttura locale

```text
data/
  raw/
    wikipedia-it/
      itwiki-YYYYMMDD-pages-articles.xml.bz2
      source.json
  clean/
    italiano-v1/
      documents.jsonl
      manifest.json
  derived/
    italiano-v1/
      tokenizer-input/
        part-000.txt
        part-001.txt
```

`raw` e `clean` non vengono mai modificati a mano. `derived` e' una vista
rigenerabile di `clean`: il trainer riceve soltanto i file `part-*.txt`.

`documents.jsonl` conserva un documento per riga. Un record avra' questa forma:

```json
{"id":"wikipedia:123", "source":"wikipedia-it", "license":"CC BY-SA", "url":"https://it.wikipedia.org/?curid=123", "text":"Testo pulito..."}
```

`manifest.json` non contiene il corpus, ma descrive esattamente come e' stato
prodotto: fonti, URL, licenza, data e checksum del download, versione della
pulizia, filtri, conteggi e comando di training. E' cio' che rende un esperimento
ripetibile.

## Pipeline implementata

1. Scaricare e verificare il dump originale in `data/raw`.
2. Estrarre solo le pagine principali dall'XML, rimuovendo markup MediaWiki.
3. Scartare redirezioni, pagine troppo corte e testi puliti duplicati.
4. Salvare i documenti puliti in JSONL insieme alla loro provenienza.
5. Generare file testuali ordinati in `data/derived`.
6. Addestrare il tokenizer e salvare un manifesto dell'artefatto.

Non normalizziamo accenti, apostrofi o punteggiatura: sono informazione utile per
un tokenizer byte-level. Le decisioni di pulizia eliminano markup e rumore, non
riscrivono arbitrariamente l'italiano.

Il corpus definitivo di questa fase si chiamera' `italiano-wikipedia-v1`: significa
una configurazione precisa, non “l'ultima Wikipedia disponibile”. Un corpus futuro
che aggiunge Wikisource o cambia la pulizia sara' una nuova versione, per esempio
`italiano-v2`, e non sovrascrivera' questa.

## Procedura completa: `italiano-wikipedia-v1`

### 0. Costruire il trainer

```sh
cmake --preset debug
cmake --build --preset debug
```

### 1. Controllare il dump scaricato

Il downloader deve avere creato questi due file:

```text
data/raw/wikipedia-it/
  itwiki-YYYYMMDD-pages-articles.xml.bz2
  source.json
```

`source.json` contiene URL, data e checksum del dump. Non modificare il file
`.xml.bz2`: e' la sorgente immutabile da cui potremo rigenerare tutto il corpus.

### 2. Creare il corpus v1 completo

```sh
python3 utils/corpus/extract_wikipedia.py \
  --name italiano-wikipedia-v1 \
  --part-size-mib 256
```

L'estrattore legge il dump bzip2 in streaming: non lo scompatta mai interamente.
Il parsing XML e la scrittura restano ordinati, mentre la pulizia delle pagine usa
in parallelo tutti i core CPU meno uno. Per scegliere esplicitamente il numero di
processi usa, ad esempio, `--workers 4`; il corpus prodotto non cambia.

Durante l'esecuzione compare una barra con percentuale, velocita' ed ETA. La
percentuale misura i byte compressi gia' letti dal dump: e' un indicatore
dell'avanzamento della lettura, non un conteggio anticipato delle pagine. La
scrittura degli ultimi documenti puo' quindi concludersi poco dopo il 100%.

Scrive invece due rappresentazioni del testo pulito:

```text
data/clean/italiano-wikipedia-v1/
  documents.jsonl
  manifest.json

data/derived/italiano-wikipedia-v1/tokenizer-input/
  part-000.txt
  part-001.txt
  ...
```

`documents.jsonl` conserva una riga JSON per pagina con ID, titolo, URL, licenza e
testo. I file `part-*.txt` contengono solo testo e sono l'unico input del trainer.
La dimensione delle parti non cambia il vocabolario: serve solo a non creare un
singolo file troppo grande.

L'estrattore rimuove commenti, riferimenti, template annidati, tabelle, tag HTML,
categorie e sintassi di link, mantenendo il testo visibile. Elimina inoltre i
duplicati esatti del testo pulito con SHA-256 e una tabella SQLite temporanea.
E' una pulizia
deterministica `wikitext-basic-v1`, non un renderer completo di MediaWiki: questa
scelta e' registrata nel manifesto e potra' essere migliorata in un corpus v2.

Per sicurezza, l'utility rifiuta una destinazione gia' popolata. Se un'esecuzione
viene interrotta, restano file `.part`; non riavviarla sullo stesso nome. Ispeziona
o elimina manualmente quella destinazione incompleta, oppure usa un nuovo `--name`.

### 3. Addestrare il vocabolario reale a 32k

Lo script seguente legge le parti indicate dal manifesto, esegue `llm-lab` e crea
sia il modello sia il suo manifesto di provenienza:

```sh
python3 utils/corpus/train_tokenizer.py \
  --corpus-manifest data/clean/italiano-wikipedia-v1/manifest.json \
  --trainer build/debug/llm-lab \
  --output artifacts/tokenizers/italiano-wikipedia-v1.llmtok \
  --vocab-size 32000
```

Il trainer C mostra l'avanzamento della lettura, dell'indicizzazione e delle
merge BPE. Per le merge, il totale e' 32000 - 256 = 31744: ogni passo creato
aggiunge un token al vocabolario.

Il risultato e':

```text
artifacts/tokenizers/
  italiano-wikipedia-v1.llmtok
  italiano-wikipedia-v1.llmtok.json
```

Il file `.json` registra corpus, parti usate, checksum del corpus, comando,
vocabolario richiesto, numero effettivo di merge e SHA-256 del modello. Non
sovrascriviamo un artefatto esistente: un nuovo training deve avere un nuovo nome
o una scelta esplicita dell'utente.

Il target 32k e' un massimo: il trainer puo' fermarsi prima solo se il corpus non
contiene piu' coppie adiacenti da fondere. Il trainer incrementale e le sue misure
di memoria sono descritti nella sezione 5.7 di [tokenizer.md](tokenizer.md).

### 4. Usare il modello

Apri il menu interattivo e carica il file prodotto:

```sh
./build/debug/tokenizer_experiment
```

Prova frasi italiane, osserva gli ID e verifica che la decodifica restituisca gli
stessi byte di partenza.

## Download del dump di Wikipedia

L'utility [../utils/corpus/download_wikipedia.py](../utils/corpus/download_wikipedia.py)
usa solo la libreria standard di Python 3.8 o successivo. Scarica il dump `pages-articles`, legge
l'MD5 pubblicato da Wikimedia nel file `md5sums.txt` della stessa versione e lo
verifica prima di rendere disponibile il file definitivo. Scrive anche `source.json`
con URL, data, MD5 della fonte e SHA-256 calcolato localmente. L'MD5 serve a
controllare il trasferimento rispetto alla fonte; lo SHA-256 identifica con piu'
precisione il file che abbiamo effettivamente usato.

L'URL iniziale contiene `latest`, ma la lista dei checksum usa il nome datato del
dump. L'utility lo risolve prima del download e salva il file con quel nome datato:
in questo modo il manifesto punta a uno snapshot preciso, non a un "latest" che
cambia nel tempo. Il file viene inoltre scaricato dalla directory datata dello
snapshot, non dall'alias `latest`.

Da macOS o Linux:

```sh
python3 utils/corpus/download_wikipedia.py
```

Da Windows PowerShell:

```powershell
py utils/corpus/download_wikipedia.py
```

Il download occupa vari GiB e non viene eseguito automaticamente dal progetto.
L'utility accetta `--output-dir` e `--url` per scegliere una cartella o un dump
specifico. Per controllare fonte e checksum senza scaricare il dump:

```sh
python3 utils/corpus/download_wikipedia.py --check
```

`--help` mostra tutte le opzioni.

Il prossimo componente sara' un estrattore: leggere l'XML compresso e produrre
`documents.jsonl`. Lo terremo distinto dal downloader, cosi' ogni passaggio e'
facile da capire e ripetere.
