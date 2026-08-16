# Dati e tokenizzazione

## 1. Il corpus

Il corpus e' l'insieme dei testi da cui il modello impara. Prima dell'addestramento servono operazioni come:

- raccolta con fonti e licenze appropriate;
- deduplicazione;
- rimozione di testo corrotto, spam e dati sensibili;
- normalizzazione del formato e della codifica;
- suddivisione in training, validation e test set.

La qualita' dei dati influenza direttamente qualita', copertura, sicurezza e bias del modello. Per un progetto didattico useremo un corpus piccolo, lecito e facilmente ispezionabile.

Il corpus non e' quindi testo messo insieme a caso: la sua composizione determina
quali parole, registri e argomenti il modello incontrera' piu' spesso. Il piano
concreto di raccolta e preparazione del corpus italiano e' in
[docs/corpus.md](../docs/corpus.md).

### Perche' una fonte sola non basta

Wikipedia e' pulita e legalmente limpida, e per questo e' il punto di partenza
naturale. Ma e' scritta tutta nello stesso modo: terza persona, tempo presente,
frasi dichiarative, strutture ripetute. Un modello che ha letto solo quello
scrive in quel modo qualunque cosa gli si chieda, perche' non ha mai visto un
dialogo, un racconto o una discussione.

Il rimedio non e' piu' testo: e' testo **diverso**. Aggiungere pagine web porta
la lingua contemporanea e i registri informali; aggiungere libri di pubblico
dominio — Wikisource e Project Gutenberg — porta la prosa lunga, che nessun'altra
fonte contiene: un romanzo ha periodi, dialoghi e una struttura che una voce
enciclopedica non ha mai.

Mescolare fonti introduce pero' un problema che non esiste con una sola: lo
stesso testo puo' comparire due volte. Wikipedia e' copiata ovunque nel web,
quindi un corpus web ne contiene inevitabilmente delle copie. Se una copia
finisce nell'insieme di addestramento e l'altra in quello di valutazione, il
modello viene misurato su testo che ha gia' letto e il risultato sembra migliore
di quanto sia. La **deduplicazione incrociata** serve a questo, e va fatta prima
di dividere i dati, non dopo.

La composizione del corpus multi-sorgente e' specificata in
[docs/corpus-multi-sorgente.md](../docs/corpus-multi-sorgente.md).

## 2. Dal testo ai token

Una rete neurale riceve numeri, non caratteri. Il tokenizer spezza una stringa in unita' discrete dette **token** e assegna a ogni token un ID intero.

```text
"Ciao mondo!" -> ["Ciao", " mondo", "!"] -> [1542, 9381, 0]
```

Spesso i token non coincidono con le parole: possono essere caratteri, byte, parole intere o sotto-parole.

| Strategia | Vantaggio | Limite |
|---|---|---|
| Caratteri | Molto semplice, nessun token sconosciuto | Sequenze lunghe |
| Parole | Intuitiva | Vocabolario enorme, parole ignote |
| Byte | Copre qualunque testo | Sequenze ancora relativamente lunghe |
| BPE / subword | Buon compromesso; comune nei moderni LLM | Richiede addestrare o usare un tokenizer |

Per iniziare e' utile un tokenizer a caratteri o byte; in seguito passeremo a BPE per capire il compromesso fra vocabolario e lunghezza della sequenza.

### Quanto grande deve essere il vocabolario?

Un vocabolario non e' gratuito. Aggiungere token BPE accorcia spesso le sequenze,
ma aumenta il numero di righe della matrice di embedding e dell'output del modello.

| Vocabolario piccolo | Vocabolario grande |
|---|---|
| Parole spezzate in piu' token | Sequenze frequenti rappresentate da meno token |
| Frasi piu' lunghe per il Transformer | Frasi piu' corte per il Transformer |
| Meno memoria e meno parametri nelle tabelle del modello | Piu' memoria e piu' parametri nelle tabelle del modello |
| Training BPE piu' rapido | Training BPE piu' lungo e vocabolario piu' specifico al corpus |

Per esempio, a seconda delle merge apprese:

```text
vocabolario piccolo: "costruzione" -> ["costru", "zione"]
vocabolario grande:  "costruzione" -> ["costruzione"]
```

Non esiste una dimensione migliore in assoluto. Conta il corpus, il linguaggio,
la memoria disponibile e la dimensione del Transformer che costruiremo dopo.
Un token raro che occupa una riga nel vocabolario non e' automaticamente utile.

Per studiare le merge possiamo usare 4k--8k token. Per il primo tokenizer italiano
da usare davvero, 32k e' un obiettivo ragionevole: la scelta finale dipende dalla
copertura del corpus e dalle misure, non dal computer su cui il progetto viene
sviluppato.

## 3. Token, ID ed embedding: tre cose diverse

Questi tre termini sono collegati, ma non sono la stessa cosa:

| Concetto | Esempio | A cosa serve |
|---|---|---|
| Token | La sequenza di byte che forma `"ciao"` | E' l'unita' di testo riconosciuta dal tokenizer. |
| ID | `258` | E' l'indirizzo numerico del token nel vocabolario. |
| Embedding | `embedding[258]` | E' il vettore numerico che ricevera' il Transformer. |

L'ID non possiede un significato linguistico da solo. Significa semplicemente: “vai alla posizione 258 del vocabolario”.

### Un esempio completo

Prendiamo `"ciao"`. All'inizio, in UTF-8, ogni carattere ASCII occupa un byte:

```text
"ciao" -> byte [99, 105, 97, 111]
        -> ID base [99, 105, 97, 111]
```

Se BPE trova spesso la coppia `c` + `i`, puo' creare il primo nuovo token:

```text
merge(99, 105) -> ID 256 -> token "ci"
```

Se poi trova spesso `"ci"` + `a` e infine `"cia"` + `o`:

```text
ID 257 -> token "cia"
ID 258 -> token "ciao"
```

Dopo il training, il tokenizer puo' quindi produrre:

```text
"ciao" -> [258]
```

Il numero `258` non e' un codice Unicode ne' il numero associato alla parola in italiano: e' solo l'ID assegnato a quella voce del nostro vocabolario.

Per un carattere accentato, un byte e un carattere non coincidono sempre. Per esempio `è` e' il carattere Unicode `U+00E8`, ma in UTF-8 usa due byte:

```text
"è" -> byte [195, 168] -> ID base [195, 168]
```

BPE puo' anche fondere quei due byte in un solo ID nuovo se la sequenza e' frequente.

Nel modello futuro il passaggio finale sara':

```text
ID 258 -> embedding[258] -> vettore numerico usato dal Transformer
```

## 4. Vocabolario e token speciali

Il **vocabolario** e' la tabella `token <-> ID`. Include spesso token speciali:

- `BOS` (inizio sequenza);
- `EOS` (fine sequenza);
- `PAD` (padding per allineare le lunghezze);
- eventualmente `UNK` (token sconosciuto).

Un tokenizer deve essere reversibile quanto possibile: `decode(encode(testo))` dovrebbe restituire lo stesso testo o una versione normalizzata prevedibile.

Nel progetto il vocabolario e' 32.000 token imparati dal corpus, piu' `<EOD>` che
segna la fine di un documento, piu' **sette identificatori riservati e vuoti**.

Quei sette slot sembrano uno spreco e invece sono una precauzione necessaria. Il
vocabolario e' l'unica dimensione del modello che non si puo' cambiare dopo
l'addestramento: le tabelle di embedding e di uscita hanno una riga per ogni
identificatore, quindi aggiungerne uno significa cambiare forma a quelle matrici
e buttare via i pesi. Se in futuro si vorra' insegnare al modello a conversare
serviranno marcatori come `<|user|>` e `<|assistant|>`: riservarli adesso costa
un migliaio di parametri, aggiungerli dopo costerebbe l'intero addestramento.

Uno slot riservato non compare mai nei dati. Riceve gradiente solo perche'
partecipa al denominatore della softmax, che ne tiene bassa la probabilita': il
modello impara a non produrlo mai, ed e' esattamente cio' che serve finche' non
gli si dara' un significato.

## 5. Creare esempi per il language modeling

Da una sequenza di ID tagliamo finestre di lunghezza `L`. Gli input sono tutti i token tranne l'ultimo; i target sono la stessa sequenza spostata di una posizione.

```text
sequenza: [A, B, C, D, E]
input:    [A, B, C, D]
target:   [B, C, D, E]
```

Il modello deve quindi imparare `A -> B`, `A B -> C`, e cosi' via.

Per i dettagli su come il nostro progetto salva e ricostruisce queste associazioni, consulta [la specifica del tokenizer](../docs/tokenizer.md).

Prosegui con [l'architettura Transformer](03-architettura-transformer.md).
