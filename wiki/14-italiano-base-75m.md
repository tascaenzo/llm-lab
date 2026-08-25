# 14 — Italiano-Base-75M: dal laboratorio a un modello utile

Il Modello Minimal ha risposto alla domanda piu' importante della prima fase:
il progetto sa addestrare una rete autoregressiva completa su CPU e Metal.
`Italiano-Base-75M` e' il passo successivo: non cambia idea di modello, ma da'
alla stessa architettura abbastanza profondita', contesto e capacita' da rendere
utili i completamenti in italiano.

La specifica operativa e' in
[Italiano-Base-75M](../docs/italiano-base-75m.md).

## Su quali dati

Il modello nasceva per essere addestrato sulla sola Wikipedia italiana. La
scelta e' cambiata: il corpus diventa multi-sorgente, perche' Wikipedia insegna
un registro solo e l'obiettivo dichiarato include il testo narrativo, che li'
non c'e'.

Il momento per decidere e' adesso e non dopo, per un motivo preciso: il
tokenizer si costruisce sul corpus, e il vocabolario che ne esce non si puo'
piu' cambiare una volta addestrato il modello. La composizione delle fonti e' in
[docs/corpus-multi-sorgente.md](../docs/corpus-multi-sorgente.md).

## Perche' non basta rendere il Modello Minimal piu' lungo

Il riferimento attuale ha un layer, una head, hidden size 64 e contesto 32. E'
ottimo per verificare un errore in attention, nel backward o nei checkpoint,
ma puo' rappresentare pochissime regolarita' del linguaggio e ricorda soltanto
una finestra molto corta.

Un completamento italiano utile richiede di combinare informazioni diverse:

- accordi grammaticali e punteggiatura;
- soggetto, verbo e oggetti anche lontani nella frase;
- significato locale delle parole;
- struttura di un paragrafo;
- registro del prompt.

Per questo il modello grande usa piu' blocchi Transformer, piu' teste di
attention e un MLP SwiGLU in ogni blocco.

## La configurazione scelta

```text
12 blocchi
hidden size 512
8 head di attention
SwiGLU 1536
contesto 512 token
circa 74 milioni di parametri
```

Non e' la configurazione piu' grande che potrebbe teoricamente entrare nella
memoria del Mac. E' una scelta bilanciata: lo split di training contiene circa
1,56 miliardi di token, cioe' all'incirca 21 token per parametro. Rendere il
modello molto piu' grande lo farebbe allenare con meno esempi per ogni peso e
rallenterebbe molto gli esperimenti senza una garanzia di testo migliore.

## Cosa cambia in un blocco Transformer

Ogni blocco applica due trasformazioni, ognuna seguita dalla sua connessione
residua:

```text
stato
  -> RMSNorm -> attenzione causale multi-head -> proiezione -> + stato
  -> RMSNorm -> SwiGLU                         -> proiezione -> + stato
```

L'attenzione decide da quali token precedenti raccogliere informazioni.
L'MLP SwiGLU trasforma queste informazioni posizione per posizione: e' una
parte sostanziale della capacita' del modello, non un dettaglio opzionale.
Ripetere il blocco 12 volte permette agli stati di diventare progressivamente
piu' astratti.

## Training: un passo deve significare qualcosa

In un modello piccolo era accettabile scegliere finestre casuali dal corpus.
Con miliardi di token, scegliere con rimpiazzo rende difficile sapere quanto
testo e' stato davvero visto: alcune finestre ricompaiono e altre non vengono
mai selezionate.

Il trainer del modello 75M usa quindi epoche riproducibili, con
campionamento streaming senza rimpiazzo. Un'epoca e' il passaggio sui circa
1,56 miliardi di token train. Con 4.096 token effettivi per update equivale a
circa 381 mila update.

Il trainer implementa accumulo dei gradienti, warmup lineare, cosine decay,
checkpoint atomico periodico, ripresa dello stato, campionamento streaming
senza rimpiazzo e clipping a norma globale. Warmup e decay non rendono il
modello piu' intelligente da soli, ma gli permettono di imparare in modo
stabile per centinaia di migliaia di update.

## Completamento e dialogo sono due prodotti

Un modello preaddestrato su Wikipedia puo' imparare a proseguire:

```text
La fotosintesi clorofilliana e' il processo mediante il quale...
```

ma non riceve esempi sufficienti del tipo:

```text
Utente: Come posso cuocere il riso?
Assistente: Per cuocere il riso puoi...
```

Quindi il percorso e' diviso in due checkpoint:

```text
testi italiani generali
        -> Italiano-Base-75M
        -> completamento e proseguimento del testo

dialoghi istruzione/risposta con licenza chiara
        -> fine-tuning supervisionato
        -> Italiano-Chat-75M
        -> risposte nel formato conversazionale
```

Nel fine-tuning il modello deve ricevere token di ruolo espliciti e calcolare
la loss soltanto sulle parole dell'assistente. Senza questa regola, imparerebbe
anche a generare il testo del prompt invece di concentrarsi sulla risposta.

## Cosa aspettarsi onestamente

`Italiano-Base-75M` puo' diventare un piccolo modello locale interessante per
completare frasi, titoli, descrizioni e paragrafi brevi. Non sara' comparabile
a un LLM commerciale molto piu' grande, ne' dovra' essere usato per consulenza
medica, legale o finanziaria.

Il suo valore e' avere un percorso completo e osservabile: dati con
provenienza, runtime proprietario, training locale, checkpoint verificabile e
una misura chiara della qualita'.
