# Dataset autoregressivo

## Dal documento all'esempio di training

Il corpus pulito contiene documenti, non coppie domanda-risposta. Nel language
modeling autoregressivo la risposta corretta e' gia' nel testo: per ogni posizione
il target e' il token immediatamente successivo.

```text
sequenza: [A, B, C, D, E]
input:    [A, B, C, D]
target:   [B, C, D, E]
```

Non annotiamo quindi manualmente i dati e non raggruppiamo le pagine per tema.
Trasformiamo il testo esistente in finestre contigue e lasciamo al modello il
compito di apprendere le regolarita' statistiche.

## Split per documento

Ogni pagina Wikipedia viene assegnata interamente a uno dei tre split:

- training: aggiorna i pesi del modello;
- validation: misura il comportamento durante lo sviluppo;
- test: viene usato soltanto per la valutazione finale.

L'assegnazione usa l'ID stabile del documento. Lo stesso ID e la stessa versione
della regola producono sempre lo stesso split, indipendentemente dall'ordine delle
righe. La configurazione iniziale e' 90% training, 5% validation e 5% test.

Lo split avviene prima della tokenizzazione. Non dividiamo una pagina in frammenti
appartenenti a set diversi: farlo permetterebbe al modello di vedere nel training
una parte dello stesso testo usato per valutarlo.

Questa non e' una separazione per argomento. Tutti gli split devono conservare,
per quanto possibile, la varieta' del corpus. Uno split tematico avrebbe un altro
scopo: misurare esplicitamente la generalizzazione verso un dominio escluso dal
training.

## Confine tra documenti

Il tokenizer BPE rappresenta soltanto byte del testo e possiede ID da `0` a
`tokenizer_vocabulary_size - 1`. Il dataset aggiunge un token di controllo
`<EOD>` (*end of document*) con ID uguale alla dimensione del vocabolario del
tokenizer.

Con il tokenizer italiano da 32.000 elementi:

```text
token testuali: 0..31999
<EOD>:          32000
vocabolario LM: 32001
```

`<EOD>` non viene passato a `tokenizer_decode`: durante la generazione indica che
il documento e' terminato. Inserirlo dopo ogni pagina evita di presentare la prima
parola della pagina successiva come continuazione naturale della precedente.

## Finestre e batch

Ogni esempio richiede `context_length + 1` token. I primi `context_length`
formano l'input; gli ultimi `context_length`, a partire dalla posizione seguente,
formano il target.

```text
stream:  [11, 22, 33, 44, 55]
input:   [11, 22, 33, 44]
target:  [22, 33, 44, 55]
```

Un batch contiene piu' finestre della stessa lunghezza. Il batcher sceglie gli
offset con un generatore pseudo-casuale deterministico: stesso dataset, stesso seed
e stesso stato producono gli stessi batch.

Le finestre possono includere `<EOD>` e proseguire con il documento successivo.
Questo e' intenzionale: il modello osserva il confine esplicito anziche' una
concatenazione invisibile.

## Contaminazione del tokenizer

Il tokenizer `italiano-wikipedia-v1` e' stato addestrato sull'intero corpus prima
di introdurre questi split. I pesi del language model non vedono validation e test,
ma il vocabolario BPE ne ha osservato le frequenze. Per il primo prototipo accettiamo
e registriamo questa limitazione. Un esperimento rigoroso dovra' creare prima lo
split e addestrare una nuova versione del tokenizer soltanto sui documenti di
training.

La rappresentazione binaria, le invarianti e le API sono specificate in
[docs/dataset.md](../docs/dataset.md).
