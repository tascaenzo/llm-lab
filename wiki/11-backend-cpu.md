# Backend CPU: usare davvero il processore

Il runtime definisce **quali operazioni** possono essere richieste. Il backend
CPU decide **come eseguirle** usando memoria, core e istruzioni del processore.

Questa pagina spiega il progetto senza entrare nei dettagli delle API C. La
specifica implementativa e' in
[docs/backend-cpu.md](../docs/backend-cpu.md).

## 1. Il ruolo del backend CPU

Quando il modello chiede una moltiplicazione tra matrici, non deve creare thread
o conoscere il tipo di processore. Il percorso e' questo:

```text
modello -> operazione del runtime -> backend CPU -> kernel CPU -> processore
```

Il backend CPU ha quattro responsabilita':

- gestire le risorse appartenenti alla CPU;
- scegliere l'implementazione adatta di ogni operazione;
- distribuire il lavoro tra i core disponibili;
- offrire al runtime lo stesso contratto che useranno Metal e CUDA.

Il **kernel** svolge il calcolo numerico. L'**executor** organizza invece dove e
quando eseguire le parti di quel calcolo. Sono componenti collegati, ma non sono
la stessa cosa.

## 2. Versione di riferimento e versione veloce

Per ogni operazione importante conviene conservare un kernel di riferimento:
un ciclo C semplice, facile da verificare e normalmente eseguito da un solo
thread. Questo kernel e' la risposta leggibile alla domanda “qual e' il calcolo
matematico?”.

Accanto ad esso possono esistere implementazioni piu' veloci:

```text
matmul
  -> kernel di riferimento
  -> kernel a blocchi e multithread
  -> libreria BLAS, se abilitata
```

La versione veloce non sostituisce il riferimento. I test possono confrontarle
sugli stessi input e individuare errori introdotti dalle ottimizzazioni.

## 3. Core, thread e parallelismo

Un processore moderno contiene piu' core. Un programma deve creare lavoro
eseguibile in parallelo per usarli contemporaneamente.

Il backend non deve creare e distruggere thread per ogni somma o matmul: il
costo potrebbe superare quello del calcolo. Creera' invece un piccolo gruppo di
thread persistenti, chiamato **thread pool**, alla nascita del backend.

```text
operazione grande
   |
   v
executor divide il lavoro
   |------- parte 1 -> worker 1
   |------- parte 2 -> worker 2
   |------- parte 3 -> worker 3
   `------- parte 4 -> worker 4
                       |
                       v
                 risultato completo
```

Il thread che ha richiesto l'operazione partecipa al lavoro. Per un'operazione
piccola, il backend usa un solo thread: coordinare molti worker avrebbe un costo
inutile.

## 4. Come si divide il lavoro

Non tutte le operazioni si parallelizzano nello stesso modo.

| Operazione | Divisione naturale |
|---|---|
| add, multiply, scale | intervalli indipendenti di elementi |
| riduzioni | righe indipendenti; risultato locale per ogni riga |
| matmul | blocchi distinti della matrice di uscita |
| gather | gruppi di indici in uscita |
| softmax | righe indipendenti |
| cross-entropy | righe, poi somma controllata delle loss |

`scatter_add` richiede piu' attenzione: due indici possono indicare la stessa
riga e quindi due thread potrebbero scrivere nello stesso punto. La versione
attuale resta seriale. Una futura versione parallela dovra'
usare partizioni, buffer locali o sincronizzazione esplicita.

## 5. SIMD: piu' numeri con una istruzione

Oltre a usare piu' core, ogni core puo' elaborare piu' valori insieme tramite
istruzioni vettoriali, dette **SIMD**.

```text
scalare:  una istruzione -> un valore
SIMD:     una istruzione -> piu' valori
```

Le famiglie di processori espongono istruzioni diverse, per esempio NEON su
ARM e AVX su alcune CPU x86. Il backend deve rilevare le capacita' disponibili
e scegliere un kernel compatibile. Deve sempre esistere un percorso C portabile.

Multithreading e SIMD risolvono problemi differenti e possono collaborare:
piu' thread lavorano su parti diverse, mentre ogni thread elabora piccoli
vettori di numeri.

## 6. Cache e matmul a blocchi

La CPU e' molto piu' veloce quando riusa dati gia' presenti nelle cache vicine
ai core. Una moltiplicazione matriciale ingenua puo' rileggere continuamente
dati lontani in memoria.

Il **tiling** divide le matrici in blocchi abbastanza piccoli da essere
riutilizzati nella cache:

```text
matrici grandi -> piccoli blocchi -> calcolo e riuso -> blocco di output
```

Per un language model, la matmul e' una delle operazioni piu' importanti. Dopo
la correttezza, il suo ordine dei cicli, il tiling, SIMD e la distribuzione dei
blocchi tra thread avranno un impatto molto maggiore di piccole ottimizzazioni
su codice usato raramente.

## 7. Determinismo e numeri in virgola mobile

L'addizione in virgola mobile non e' perfettamente associativa. Cambiare
l'ordine delle somme puo' produrre differenze minime:

```text
(a + b) + c  puo' differire leggermente da  a + (b + c)
```

Il backend offrira' una modalita' deterministica per test e debugging. Le
versioni parallele delle riduzioni useranno un ordine definito di combinazione.
I risultati vengono confrontati con tolleranze dichiarate, non sempre bit per
bit.

## 8. Perche' questa fase viene prima dell'autograd

L'autograd eseguira' molte delle stesse operazioni sia nel forward sia nel
backward. Costruirlo sopra una base priva di una strategia CPU renderebbe piu'
difficile separare problemi matematici da problemi di esecuzione.

L'ordine deciso e' quindi:

1. kernel CPU corretti di riferimento;
2. struttura separata del backend;
3. executor e parallelizzazione controllata;
4. benchmark e confronto con il riferimento;
5. primi kernel SIMD e matmul a blocchi;
6. autograd e layer neurali.

Non serve rendere ogni operazione perfetta prima di creare il modello. Serve
pero' un backend con confini stabili, testabile e gia' capace di usare piu' core.

## 9. Come valuteremo il backend

La correttezza viene prima della velocita'. Ogni incremento deve rispondere a
tre domande:

- il risultato coincide con il kernel di riferimento entro la tolleranza?
- il codice resta corretto con uno o piu' thread?
- il tempo misurato migliora su dimensioni realistiche?

I test unitari verificano la correttezza. I benchmark misurano il tempo e non
devono fallire solo perche' una macchina e' temporaneamente lenta.

## 10. Stato attuale

Sono implementati il backend CPU sincrono, l'allocazione allineata e i kernel C
di riferimento per operazioni elementwise, riduzioni, matmul, gather/scatter,
softmax e cross-entropy. Il codice specifico e' isolato in
`src/runtime/backends/cpu/`.

Sono implementati anche contesto configurabile, rilevamento dei processori,
thread pool persistente e `parallel_for`. Memoria di grandi dimensioni,
elementwise, riduzioni per riga, matmul, gather, softmax e backward della
cross-entropy usano il parallelismo oltre soglie conservative. La matmul usa
blocchi per la cache. L'executor assegna i chunk con un contatore atomico, cosi'
i worker non devono contendersi un mutex per ogni porzione di lavoro.

La suite benchmark misura le primitive principali con liste come
`1,2,4,8,auto`, calcola speedup ed efficienza e produce risultati JSON
confrontabili con una baseline della stessa macchina.

`scatter_add` e la cross-entropy forward restano seriali per evitare race o
riduzioni non controllate. I percorsi piu' caldi usano gia' SIMD baseline NEON
su ARM64 e SSE2 su x86-64. Il prossimo lavoro prestazionale e' il dispatch per
estensioni piu' avanzate e, se misurata utile, una BLAS opzionale; il core
neurale puo' ora iniziare sopra un backend gia' parallelo.
