# Corpus multi-sorgente — specifica

**Stato:** progettato, non implementato. Estende il [piano operativo del
corpus](corpus.md) da una sola fonte a un insieme di fonti con licenza
verificata, deduplicazione incrociata e quote di miscelazione dichiarate.

Il corpus prodotto si chiama `italiano-v3` e sostituisce
`italiano-wikipedia-v1` come base di addestramento di
[Italiano-Base-75M](italiano-base-75m.md).

## Perche' non basta Wikipedia

Wikipedia e' insolitamente pulita e resta il nucleo del corpus, ma e' un
registro solo: terza persona, dichiarativa, presente, fortemente templata. Un
modello addestrato unicamente su di essa produce quel registro qualunque sia il
prompt.

L'obiettivo di prodotto dichiarato chiede pero' di *«proseguire testo
informativo e narrativo»*, e il narrativo in Wikipedia e' quasi assente. Il
passo successivo, `Italiano-Chat-75M`, richiede struttura dialogica, che in
Wikipedia manca del tutto.

C'e' inoltre un vincolo di sequenza che rende questa decisione urgente: il
tokenizer determina embedding e output head, quindi **cambiarlo dopo il
pretraining significa buttare via il modello**. Il vocabolario va deciso sul
corpus definitivo, prima del run lungo.

## Obiettivo

| Campo | Valore | Motivazione |
|---|---:|---|
| token split train | circa 3 miliardi | 40 token per parametro sul 75M |
| lingua | italiano | monolingue, come il tokenizer |
| vocabolario tokenizer | 32.000 | invariato: allargarlo sposta capacita' dai layer alle matrici lessicali |
| slot speciali riservati | 8 | `<EOD>` piu' sette slot liberi per i token di ruolo |

Il 75M a 3 miliardi di token e' deliberatamente oltre l'ottimo di Chinchilla.
E' la scelta di Llama: un modello piccolo addestrato a lungo costa meno per
sempre in inferenza. Crescere a 150M resta possibile in seguito riusando il
100% dei pesi, come descritto in [Italiano-Base-75M](italiano-base-75m.md).

## Fonti e quote

### Dipendenze

Il progetto non ha dipendenze Python esterne, ma FineWeb-2 e' distribuito in
Parquet e la stdlib non lo legge. La linea si sposta quindi da *«nessuna
dipendenza»* a **«nessuna dipendenza a runtime, strumenti offline liberi»**:
`pyarrow` e' richiesto solo da `normalize_source.py`, che gira una volta in fase
di preparazione dati. Il download resta stdlib pura, e il runtime di training in
C non cambia.

| Fonte | Quota | Licenza | Registro che aggiunge |
|---|---:|---|---|
| FineWeb-2, sottoinsieme `ita_Latn` | 55% | ODC-By 1.0 | web generale, lingua contemporanea, registri informali |
| Wikipedia italiano | 20% | CC BY-SA | enciclopedico, nucleo pulito |
| Wikisource IT | 8% | pubblico dominio | **narrativo**: testi trascritti e revisionati |
| Gutenberg IT | 7% | pubblico dominio | **narrativo**: romanzi e saggi integrali |
| EUR-Lex italiano, Normattiva | 10% | riuso con attribuzione | formale e giuridico |

Indirizzi, verificati il 16 agosto 2026:

| Fonte | Indirizzo | Note |
|---|---|---|
| FineWeb-2 | <https://huggingface.co/datasets/HuggingFaceFW/fineweb-2> | configurazione `ita_Latn`, 85 shard in `data/ita_Latn/train/` |
| Wikipedia IT | <https://dumps.wikimedia.org/itwiki/latest/> | `itwiki-latest-pages-articles.xml.bz2` |
| Wikisource IT | <https://dumps.wikimedia.org/itwikisource/latest/> | `itwikisource-latest-pages-articles.xml.bz2`, stesso formato di Wikipedia |
| Gutenberg IT | <https://www.gutenberg.org/browse/languages/it> | catalogo bulk in `https://www.gutenberg.org/cache/epub/feeds/` |
| Liber Liber | <https://www.liberliber.it/> | nessun bulk: licenze da verificare per titolo |
| EUR-Lex | <https://eur-lex.europa.eu/> | fuori dalla prima versione |
| Normattiva | <https://www.normattiva.it/> | fuori dalla prima versione |

Su Wikisource una precisazione che costa tempo se scoperta tardi: molte pagine
del namespace principale contengono solo direttive di trasclusione, e il testo
vero sta nel namespace `Pagina:` (`ns=108`). Il formato del dump e' quello di
Wikipedia, ma l'estrattore va adattato.

Le quote sono espresse in token dopo la deduplicazione, non in documenti ne' in
byte. FineWeb-2 arriva gia' filtrato e deduplicato al proprio interno: e' la
ragione per cui viene preferito a OSCAR grezzo, che richiederebbe di
reimplementare quel filtraggio.

Nessuna fonte entra nel corpus senza una riga nel manifesto che dichiari
origine, licenza, data di acquisizione, checksum, filtri applicati e conteggi
prodotti.

FineWeb-2 pubblica l'italiano in 85 shard da circa 4,5 GiB, 332 GiB in totale.
**Non serve scaricarle tutte**: una shard misurata rende circa 2,6 miliardi di
token, quindi la quota web sta in una sola shard. `download_fineweb.py` scarica
l'intervallo richiesto e nient'altro.

EUR-Lex, Normattiva e Liber Liber restano fuori dalla prima versione: nessuno
dei tre offre un accesso bulk, servirebbe scraping, e il costo non e'
giustificato dalla quota che coprirebbero.

Le due fonti narrative implementate hanno vincoli diversi da FineWeb-2:

**Wikisource** usa lo stesso dump MediaWiki di Wikipedia, 421 MiB, e riusa
`extract_wikipedia.py`. L'estrattore e' stato pero' parametrizzato: `id`,
`source`, `license` e `url` erano cablati su `wikipedia-it`, quindi usarlo cosi'
com'era avrebbe prodotto documenti Wikisource con identificatori nel namespace
di Wikipedia — esattamente la collisione che questa specifica vieta.

**Gutenberg** si scarica un libro alla volta dai mirror ufficiali
(`gutenberg.pglaf.org`, `aleph.gutenberg.org`), perche' `www.gutenberg.org`
limita il download automatico; il catalogo italiano arriva da gutendex.com e
conta circa 1.100 titoli. Ogni file contiene una licenza in inglese di alcune
migliaia di parole, identica in tutti i libri: lasciarla dentro significherebbe
insegnare quel boilerplate mille volte. I marcatori `*** START` e `*** END` la
delimitano, ma i file piu' vecchi chiudono con `End of Project Gutenberg's
<titolo>` senza asterischi e possono contenere **entrambe** le forme: vale la
piu' a sinistra, altrimenti fra le due resta il colophon del trascrittore. Un
libro senza marcatori viene scartato invece che ripulito a indovinare.

I libri vengono infine spezzati in sezioni di circa ottomila caratteri tagliando
su righe vuote: un romanzo intero non entra in una finestra di contesto, e la
sezione conserva l'ID del libro nel proprio per risalire alla fonte.

## Schema dei documenti

Invariato rispetto a [corpus.md](corpus.md): il formato era gia' multi-sorgente.

```json
{"id":"fineweb2:0a3f...", "source":"fineweb2-ita", "license":"ODC-By-1.0", "url":"https://...", "text":"Testo pulito..."}
```

L'`id` **deve** essere prefissato dalla fonte. Senza namespace due documenti di
fonti diverse possono collidere, e poiche' lo split e' derivato dall'ID la
collisione produce assegnazioni non deterministiche.

Corollario utile: mantenendo il prefisso `wikipedia:` invariato, i documenti
Wikipedia conservano esattamente lo split che hanno oggi.

## Deduplicazione incrociata

E' la parte che, se sbagliata, invalida la valutazione invece che soltanto
l'efficienza. FineWeb-2 e CulturaX **contengono Wikipedia**, perche' e'
mirrorata ovunque nel web. Senza deduplicazione lo stesso testo finisce sia in
train sia in validation e la loss di validation risulta ottimistica.

Due livelli, in quest'ordine:

1. **Esatta.** SHA-256 del testo normalizzato (minuscole, spazi collassati,
   punteggiatura invariata). Cattura i mirror integrali a costo trascurabile.
2. **Approssimata.** MinHash su shingle di 5 parole con indice LSH, soglia di
   Jaccard 0,8. Cattura mirror parziali, boilerplate e riformattazioni.

Quando due documenti collidono si tiene quello della fonte con priorita' piu'
alta: `wikipedia` > `libri` > `istituzionale` > `web`. La copia web viene
scartata. L'ordine degli argomenti di `deduplicate_corpus.py` **e'** l'ordine di
priorita'.

L'indice delle frasi e' un filtro di Bloom, non un insieme: decine di milioni di
hash in un `set` costerebbero gigabyte, il filtro costa circa 1,2 byte per frase
e per il corpus intero resta sotto i 200 MiB. I falsi positivi non spostano il
risultato perche' la decisione dipende da una frazione di frasi, non da una.

### Taratura della soglia

Misurata su 47.359 documenti reali di FineWeb-2 con 150 mirror iniettati, cioe'
articoli avvolti in menu, banner e footer:

| Soglia | Mirror trovati | Scarti aggiuntivi |
|---:|---:|---:|
| 0,4 | 150/150 | 1,67% |
| **0,5** | **150/150** | **1,13%** |
| 0,7 | 147/150 | 0,24% |
| 0,9 | 139/150 | 0,02% |

0,5 e' il valore scelto: e' l'ultimo che mantiene il richiamo pieno. Gli scarti
aggiuntivi non sono errori — ispezionandoli si trovano pagine-template dello
stesso sito, dominate da banner sui cookie e menu di navigazione, cioe'
duplicazione reale che la deduplicazione interna di FineWeb-2 non rimuove
perche' opera sul documento intero.

Resta un compromesso da conoscere: una pagina fatta per meta' di boilerplate
condiviso e per meta' di contenuto originale viene scartata. Se il corpus finale
risultasse povero per questo, la correzione giusta non e' alzare la soglia ma
escludere dal confronto le frasi che compaiono in moltissimi documenti.

Velocita' misurata: circa 3.300 documenti al secondo, quindi un corpus da otto
milioni di documenti si deduplica in meno di un'ora.

**La deduplicazione precede l'assegnazione degli split.** Se avviene dopo, i
quasi-duplicati sono gia' finiti da parti opposte e il danno e' fatto.

## Vocabolario e slot riservati

Il tokenizer si addestra sul corpus finale, **solo split train**. Il gate
esistente che rifiuta manifest privi di `tokenizer_input_split: train` va
mantenuto: e' cio' che impedisce la contaminazione di validation e test.

Il vocabolario del modello riserva otto identificatori oltre i 32.000 del
tokenizer:

```text
32000  <EOD>
32001  <|user|>
32002  <|assistant|>
32003  <|end|>
32004..32007  liberi
model_vocabulary_size = 32008
```

Gli slot liberi non compaiono mai nei dati di pretraining: ricevono gradiente
solo attraverso il denominatore della softmax, che li spinge verso il basso. E'
il comportamento atteso. Costano 1.024 parametri ciascuno fra embedding e
output head, cioe' nulla, e sono l'unico modo per rendere possibile il
fine-tuning conversazionale senza ridimensionare il modello.

## Strumenti da costruire

| Strumento | Ruolo |
|---|---|
| `utils/corpus/build_corpus.py` | **guida l'intera catena da CLI**: sceglie fonti e quote, salta gli stadi gia' fatti |
| `utils/corpus/download_fineweb.py` | scarica le shard richieste con checksum e `source.json` |
| `utils/corpus/download_wikisource.py` | dump MediaWiki di Wikisource, riusa il downloader di Wikipedia |
| `utils/corpus/download_gutenberg.py` | catalogo da gutendex, testi dai mirror ufficiali |
| `utils/corpus/normalize_gutenberg.py` | rimuove la licenza, spezza i libri in sezioni |
| `utils/corpus/normalize_source.py` | Parquet allo schema `documents.jsonl`, ID con namespace |
| `utils/corpus/deduplicate_corpus.py` | dedup esatta e per sovrapposizione di frasi, con priorita' di fonte |
| `utils/corpus/split_by_source.py` | raccordo: la dedup vede tutte le fonti insieme, il mixer una alla volta |
| `utils/corpus/mix_corpus.py` | applica le quote in token e scrive il manifesto finale |
| `llm-lab dataset prepare --reserved-tokens N` | riserva gli slot speciali nel vocabolario del modello |

`extract_wikipedia.py` resta invariato: diventa il normalizzatore della sola
fonte Wikipedia. `prepare_tokenizer_corpus.py` e `train_tokenizer.py` non
cambiano: lo split e' derivato da FNV-1a a 64 bit dell'ID, funzione pura che non
sa da quale fonte arrivi il documento.

### Quote e conteggio dei token

Le quote sono in token, ma il tokenizer del corpus nuovo non esiste ancora. Il
mixer usa quindi il tokenizer precedente come **metro di misura**, calibrando i
byte per token separatamente su ogni fonte. L'errore introdotto e' quello gia'
misurato, il 3,8%, e calibrando per fonte si annulla quasi del tutto.

Una fonte piu' povera della propria quota viene presa per intero e la quota
effettiva finisce nel manifesto: meglio un corpus dichiaratamente sbilanciato
che un mix falsato in silenzio.

### Esecuzione

```sh
python3 utils/corpus/build_corpus.py
```

Chiede quali fonti includere, quante shard scaricare, quanti token e con quali
quote, mostra il piano con lo spazio libero e chiede conferma prima di iniziare.
Ogni stadio dichiara i propri output: se esistono viene saltato, quindi dopo un
download interrotto basta rilanciare. La disponibilita' di pyarrow e' verificata
prima del download, non dopo.

## Budget di spazio

Circa 3 miliardi di token corrispondono a 12–14 GB di testo pulito, piu' i
download compressi e gli scarti della deduplicazione. Un `.llmdat` da 3 miliardi
di token occupa circa 12 GB con identificatori a 32 bit. Il totale sta
comodamente nei 283 GiB disponibili.

## Criteri di completamento

1. Ogni fonte ha una riga di manifesto con origine, licenza, checksum, filtri e
   conteggi.
2. La deduplicazione e' misurata: quanti documenti e quanti token sono stati
   scartati, per coppia di fonti.
3. Nessun documento di validation o test ha un quasi-duplicato in train sopra
   la soglia di Jaccard dichiarata.
4. Il tokenizer e' addestrato solo su train e il suo manifesto lo dimostra.
5. I byte per token del nuovo tokenizer sono misurati e confrontati con
   `italiano-wikipedia-v2` su un campione comune. Misura di riferimento gia'
   eseguita: il tokenizer addestrato su Wikipedia rende 2,764 byte per token sul
   proprio corpus e 2,659 su FineWeb-2, cioe' **il 3,8% di degrado**. E' meno di
   quanto ci si aspetterebbe e ridimensiona il guadagno atteso dal
   riaddestramento: la ragione forte per rifare il tokenizer resta la riserva
   degli identificatori speciali, non la compressione.
6. Le quote effettive in token sono entro il 2% di quelle dichiarate.
