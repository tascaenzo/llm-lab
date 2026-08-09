# Dati e tokenizzazione

## 1. Il corpus

Il corpus e' l'insieme dei testi da cui il modello impara. Prima dell'addestramento servono operazioni come:

- raccolta con fonti e licenze appropriate;
- deduplicazione;
- rimozione di testo corrotto, spam e dati sensibili;
- normalizzazione del formato e della codifica;
- suddivisione in training, validation e test set.

La qualita' dei dati influenza direttamente qualita', copertura, sicurezza e bias del modello. Per un progetto didattico useremo un corpus piccolo, lecito e facilmente ispezionabile.

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

## 3. Vocabolario e token speciali

Il **vocabolario** e' la tabella `token <-> ID`. Include spesso token speciali:

- `BOS` (inizio sequenza);
- `EOS` (fine sequenza);
- `PAD` (padding per allineare le lunghezze);
- eventualmente `UNK` (token sconosciuto).

Un tokenizer deve essere reversibile quanto possibile: `decode(encode(testo))` dovrebbe restituire lo stesso testo o una versione normalizzata prevedibile.

## 4. Creare esempi per il language modeling

Da una sequenza di ID tagliamo finestre di lunghezza `L`. Gli input sono tutti i token tranne l'ultimo; i target sono la stessa sequenza spostata di una posizione.

```text
sequenza: [A, B, C, D, E]
input:    [A, B, C, D]
target:   [B, C, D, E]
```

Il modello deve quindi imparare `A -> B`, `A B -> C`, e cosi' via.

Prosegui con [l'architettura Transformer](03-architettura-transformer.md).
