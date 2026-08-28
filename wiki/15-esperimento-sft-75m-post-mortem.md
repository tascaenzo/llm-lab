# 15 — Esperimento Italiano-Chat-75M: esito parziale e lezioni

Questa pagina documenta il primo ciclo completo di pretraining e supervised
fine-tuning di `Italiano-Base-75M`. Il risultato e' un **esperimento a esito
parziale**: l'infrastruttura ha funzionato e il modello ha imparato gli esempi
supervisionati, ma non ha raggiunto la generalizzazione necessaria per essere
considerato un assistente italiano affidabile.

Definirlo semplicemente un fallimento sarebbe inesatto. Definirlo un successo
qualitativo lo sarebbe altrettanto. Il valore dell'esperimento e' aver separato
con misure concrete tre problemi differenti:

- correttezza della pipeline tecnica;
- qualita' e capacita' del modello base;
- qualita' e metodo di valutazione del fine-tuning.

Le specifiche operative degli artefatti sono in
[Italiano-Base-75M](../docs/italiano-base-75m.md),
[checkpoint del base](../docs/italiano-base-75m-checkpoints.md) e
[Italiano Simple QA v2](../docs/italiano-simple-qa-v2.md).

## Obiettivo dell'esperimento

Il percorso doveva verificare se un decoder da circa 75 milioni di parametri,
preaddestrato su testo italiano, potesse diventare tramite SFT un piccolo
assistente capace di:

- rispondere a domande semplici di cultura generale;
- fornire definizioni brevi;
- seguire istruzioni pratiche;
- riscrivere una frase con il tono richiesto;
- chiedere dettagli davanti a una domanda ambigua;
- non inventare risposte per entita' inesistenti;
- mantenere nomi, numeri e argomento della domanda.

Il 75M non aveva l'obiettivo di competere con modelli commerciali. Doveva
essere un gate economico prima di investire in un modello piu' grande.

## Stato del modello base

La configurazione effettiva di `Italiano-Base-75M` e':

```text
parametri                 75.010.560
layer                     12
hidden size               512
head                      8
FFN SwiGLU                1.608
contesto                  512 token
vocabolario LM            32.008
embedding/output head     non condivisi
```

Embedding e output head occupano insieme 32.776.192 parametri, circa il 43,7%
del totale. Restano circa 42,2 milioni di parametri per blocchi Transformer e
normalizzazioni. Questa distribuzione non rende il modello scorretto, ma limita
la capacita' effettiva disponibile in un modello gia' piccolo.

Il pretraining ha elaborato quasi un'epoca dello split train, pari a circa
2,56 miliardi di token. I checkpoint principali sono:

| Step | Ruolo |
|---:|---|
| 610.000 | migliore validation, loss `2,545995` |
| 624.362 | ultimo update dell'epoca |

La loss dimostra che il modello ha imparato a prevedere il testo della propria
distribuzione di validation. Non dimostra da sola che sappia recuperare fatti,
risolvere calcoli o rispondere a istruzioni.

## Il corpus di pretraining realmente utilizzato

Il corpus `italiano-v3` non e' composto principalmente da Wikipedia. Il
[manifest](../data/clean/italiano-v3/manifest.json) registra circa:

| Fonte | Documenti | Quota stimata |
|---|---:|---:|
| FineWeb2 italiano | 1.520.562 | 61,9% |
| Wikipedia italiana | 739.554 | 25,8% |
| Wikisource italiana | 522.987 | 12,3% |

FineWeb e' utile per introdurre lingua contemporanea e domini diversi, ma un
campione del materiale locale ha mostrato anche:

- pagine commerciali e cataloghi;
- testi SEO;
- notizie e previsioni rapidamente obsolete;
- boilerplate;
- frammenti multilingue;
- grammatica o formattazione debole;
- associazioni fattuali poco dense rispetto alla lunghezza del documento.

Il problema non e' la presenza di FineWeb in se', ma il suo peso dominante e
un filtro qualitativo non abbastanza severo per la capacita' del 75M.
Wikisource aggiunge inoltre testi letterari e storici, spesso distanti
dall'italiano informativo moderno richiesto al prodotto finale.

Fatti elementari come Roma capitale e il simbolo del fosforo compaiono nel
corpus. La loro presenza, tuttavia, non garantisce che un modello piccolo li
organizzi in rappresentazioni recuperabili con un prompt breve.

## Gli esperimenti di fine-tuning

I primi corpus chat, pur facendo diminuire la loss assistant, producevano
risposte fuori tema, ripetitive o semanticamente confuse. Un sanity test su un
numero minuscolo di fatti ha poi dimostrato che:

- la serializzazione chat e' corretta;
- i token di ruolo funzionano;
- la loss mascherata sui token assistant funziona;
- il modello puo' memorizzare associazioni domanda/risposta;
- checkpoint e inferenza usano tokenizer e pesi compatibili.

Il problema non era quindi un guasto elementare della pipeline SFT.

### Italiano Simple QA v2

Il corpus controllato v2 contiene:

| Misura | Valore |
|---|---:|
| Esempi complessivi | 5.551 |
| Gruppi semantici reali | 584 |
| Train | 5.032 |
| Validation | 268 |
| Test | 251 |
| Prompt manuali esclusi dal sorgente | 529 |

La differenza tra 5.551 esempi e 584 gruppi e' essenziale. Per raggiungere la
quota nominale di cinquemila righe erano state aggiunte sei formulazioni
generiche a ogni gruppo. Questo ha aumentato la ridondanza molto piu' della
conoscenza disponibile.

Il run da 600 step ha prodotto la seguente dinamica:

| Step | Loss validation | Interpretazione |
|---:|---:|---|
| 25 | 2,3857 | primo checkpoint |
| 50 | 1,9642 | miglioramento |
| 75 | **1,8178** | migliore validation |
| 100 | 1,8592 | inizio peggioramento |
| 300 | 2,4265 | overfitting evidente |
| 600 | 2,7430 | stato finale fortemente specializzato |

Allo step 600 la loss train assistant era circa `0,0023`. Una loss train quasi
nulla insieme a validation in peggioramento e' un segnale netto di
memorizzazione.

## Risultati qualitativi

Sono stati distinti tre livelli di prova.

### 1. Concetti presenti nel training

Il checkpoint finale ha risposto correttamente a 8 domande su 10 in un piccolo
campione di concetti addestrati o loro formulazioni vicine. Tra i successi:

- capitale d'Italia;
- simbolo dell'idrogeno;
- proclamazione del Regno d'Italia;
- apertura di un'email;
- riscrittura cortese;
- richiesta dei requisiti per consigliare un computer;
- rifiuto di inventare la capitale di Lumeria;
- verifica di una carica politica corrente.

Ha fallito una definizione di browser e il calcolo `7 + 5`. Il risultato prova
che il SFT ha insegnato contenuti specifici, ma non ancora una regola generale
affidabile.

### 2. Gruppi completamente esclusi dal training

Su nove gruppi della validation mai presentati durante il training, il
checkpoint finale ha ottenuto 0 risposte pienamente corrette su 9. Ha spesso
imitato la forma della risposta attesa, sostituendo pero' l'entita' o il valore:

- Nuova Zelanda associata a Dublino;
- fosforo confuso con potassio;
- `31 + 27` completato con 67;
- Karselia riconosciuta come paese inesistente, ma rinominata Pavel;
- traduzione confusa con revisione di un testo.

Alcune risposte mostravano il comportamento generale corretto, ma senza
preservare gli slot semantici della domanda. Con un criterio rigoroso non sono
risposte utilizzabili.

### 3. Completamenti del modello base

Il checkpoint base a step 610.000 e' stato provato senza protocollo chat su sei
completamenti elementari. Non ha completato correttamente:

- capitale d'Italia;
- capitale della Nuova Zelanda;
- simbolo chimico del fosforo;
- lingua degli antichi Romani;
- somma `31 + 27`;
- definizione di browser.

Questo risultato indica che il SFT non ha distrutto una competenza fattuale
gia' solida. Il modello base non la possedeva in forma affidabile.

## Che cosa ha funzionato

L'esperimento ha validato componenti sostanziali del progetto:

- corpus e tokenizer riproducibili;
- dataset autoregressivi e SFT versionati;
- decoder Transformer completo;
- backward e AdamW;
- training su Metal e CUDA;
- accumulo dei gradienti;
- scheduler, clipping e sampler;
- checkpoint atomici e ripresa;
- loss assistant mascherata;
- token di ruolo chat;
- inferenza incrementale con KV cache;
- generazione deterministica con seed;
- capacita' di apprendere e memorizzare esempi supervisionati.

Il checkpoint base non e' corrotto e il lavoro non va eliminato. Rimane una
baseline riproducibile e il punto di partenza per esperimenti di continued
pretraining.

## Che cosa non ha funzionato

Il modello ottenuto non soddisfa il gate di assistente generale perche':

- il base non mostra conoscenza elementare recuperabile in modo affidabile;
- il mix di pretraining contiene troppo web rumoroso per un modello piccolo;
- embedding e head non condivisi assorbono una quota elevata dei parametri;
- il corpus SFT presenta molte formulazioni ma pochi concetti indipendenti;
- il modello confonde nomi, numeri e risposte appartenenti a esempi simili;
- la matematica viene trattata come associazione testuale, non come procedura;
- il checkpoint migliore per loss su fatti completamente nuovi era troppo
  precoce per aver appreso gran parte dei concetti del train;
- il checkpoint finale memorizza il train ma peggiora sui gruppi nuovi.

## Un problema emerso nella valutazione

Tenere tutte le parafrasi di un fatto nello stesso split impedisce una forma
banale di contaminazione ed e' corretto per misurare conoscenze completamente
nuove. Non e' pero' sufficiente per scegliere il checkpoint di un SFT che deve
anche generalizzare la formulazione dei fatti addestrati.

La prossima versione deve avere almeno tre suite distinte:

1. **Parafrasi note semanticamente**: il fatto e' nel train, ma la formulazione
   e' esclusa. Misura instruction-following e generalizzazione linguistica.
2. **Conoscenze nuove**: l'intero gruppo e' escluso. Misura cio' che il base sa
   gia' e quanto trasferisce tra concetti.
3. **Robustezza**: entita' inventate, domande ambigue, conservazione di nomi e
   numeri, ripetizione e deviazioni fuori tema.

Il test finale deve restare sigillato. Le loss di corpus differenti non devono
essere confrontate come se misurassero la stessa distribuzione.

## Cause probabili

I risultati non identificano una sola causa, ma sostengono una combinazione:

```text
corpus base rumoroso e sbilanciato
        +
capacita' limitata del corpo Transformer
        +
SFT ridondante
        +
selezione del checkpoint con una sola validation
        =
memorizzazione senza sufficiente generalizzazione
```

La qualita' del corpus base e' il primo elemento da correggere perche' puo'
essere verificato economicamente sul modello esistente. Non e' ancora provato
che sia l'unico collo di bottiglia.

## Piano per Italiano-Base-75M v2

Il checkpoint consigliato a step 610.000 viene conservato immutato come
`v1-experimental`. Il nuovo ramo deve effettuare continued pretraining
caricando i pesi ma creando nuovi optimizer, sampler e scheduler.

La CLI attuale permette `model train --resume`, ma la ripresa richiede lo
stesso dataset e conserva lo stato precedente. Prima del nuovo esperimento va
quindi implementata un'opzione `model train --base`, analoga a quella del
comando SFT.

Una prima composizione da validare, non ancora definitiva, e':

| Fonte | Quota iniziale |
|---|---:|
| Wikipedia italiana | 55% |
| Fonti educative, scientifiche e istituzionali | 20% |
| Web italiano severamente filtrato | 15% |
| Narrativa e saggistica selezionata | 10% |

Il lavoro sui dati deve includere deduplicazione tra fonti, controllo della
lingua, rimozione del boilerplate, filtri SEO, qualita' minima del testo e
report per fonte.

La sequenza sperimentale proposta e':

```text
Base 75M v1, step 610.000
        -> pilot identici da 10-50M token su data recipe alternative
        -> scelta tramite validation comune e suite qualitative congelate
        -> continued pretraining da 300-500M token
        -> Base 75M v2
        -> nuovo SFT controllato
```

Il nuovo SFT non deve riutilizzare i checkpoint chat attuali. Deve partire
dalla nuova base e contenere piu' gruppi semantici reali, al massimo due o tre
formulazioni per gruppo e un learning rate piu' conservativo.

## Dal 75M al 300M

Il prossimo modello importante proposto e' circa 300M, non 800M. Prima del
salto devono essere superati questi gate:

- il nuovo corpus migliora il 75M in un pilot controllato;
- il base supera una suite minima di completamenti italiani e fattuali;
- la SFT generalizza a parafrasi non addestrate;
- il budget di memoria e throughput e' misurato;
- il test finale rimane separato dalle decisioni di sviluppo.

Il 300M dovra' normalmente essere inizializzato da zero, perche' cambiare
hidden size, numero di layer o dimensioni FFN rende incompatibili molti pesi
del 75M. Verranno invece riutilizzati tokenizer, pipeline dati, filtri,
runtime, backend, suite di valutazione e lezioni sperimentali.

Prima del 300M conviene valutare:

- weight tying tra embedding e output head;
- mixed precision BF16/FP16;
- activation checkpointing;
- riduzione dei buffer temporanei;
- configurazione adatta a circa 4-6 miliardi di token puliti;
- uso di hardware esterno per il training completo.

Una crescita diretta dei pesi `75M -> 150M -> 300M` rimane una possibile linea
di ricerca, non una capacita' implementata ne' la strada raccomandata per il
primo 300M.

## Criteri di uscita per il prossimo ciclo

Prima di dichiarare riuscito un nuovo modello chat devono essere pubblicati:

- manifest e statistiche del corpus;
- configurazione e checksum del tokenizer;
- numero effettivo di token elaborati;
- curve train e validation;
- benchmark comune tra base v1 e base v2;
- risultati su parafrasi note e gruppi nuovi;
- esempi riusciti e falliti, senza selezione opportunistica;
- checkpoint best e latest con significato documentato;
- limiti noti del modello.

Il gate qualitativo deve essere deciso prima del training, non dopo aver visto
gli output.

## Conclusione

Il primo Italiano-Chat-75M non e' un assistente generale riuscito. E' pero' un
esperimento riuscito nel suo compito piu' importante: rendere osservabile il
confine tra una pipeline tecnicamente corretta e un modello qualitativamente
utile.

Il lavoro svolto non va buttato. Il 75M ha evitato di scoprire gli stessi
problemi dopo un training molto piu' costoso su 300M o 800M. I checkpoint
restano baseline e punti di partenza; il corpus e la valutazione vengono
revisionati; l'infrastruttura viene riutilizzata.

La prossima fase non consiste nel nascondere il semi-fallimento aumentando i
parametri. Consiste nel correggere dati e misure sul modello piccolo, dimostrare
il miglioramento e solo allora scalare.
