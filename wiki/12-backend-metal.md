# Backend Metal: portare i tensori sulla GPU Apple

## Che cosa fa

Il backend Metal prende le operazioni astratte del runtime, come somma, softmax
e moltiplicazione tra matrici, e le esegue sulla GPU Apple.

```text
layer neurale
    |
    v
runtime tensoriale
    |
    +-- CPU   -> thread pool e kernel C
    |
    `-- Metal -> command queue e kernel GPU
```

Il futuro Transformer non dovra' conoscere Metal. Chiamera' sempre le API dei
tensori; sara' il backend a scegliere buffer, pipeline e numero di thread GPU.

## Device, buffer, pipeline e command buffer

- Il **device** rappresenta la GPU.
- Un **buffer** contiene i dati di un tensore.
- Una **pipeline** e' un kernel compilato e pronto.
- Un **command buffer** e' lavoro ordinato inviato alla GPU.

Apple Silicon usa memoria unificata: CPU e GPU condividono la memoria fisica.
Questo riduce le copie, ma non elimina la sincronizzazione. Prima che la CPU
legga un risultato, la GPU deve aver terminato di scriverlo.

## Perche' esiste un pool di memoria

Creare e distruggere continuamente grandi buffer costa tempo. Quando un tensore
viene distrutto, il backend puo' conservare il suo buffer e riutilizzarlo per un
tensore successivo di dimensione compatibile.

```text
tensore distrutto -> buffer in cache -> nuovo tensore -> stesso buffer riusato
```

Il pool ha limiti precisi, quindi non cresce senza controllo.

## Esecuzione normale e batch

Normalmente una chiamata attende il proprio risultato. E' semplice da capire e
utile per testare.

In un batch, invece, piu' operazioni vengono registrate nello stesso pacchetto
di lavoro. La GPU le esegue in ordine e il programma invia e attende soltanto
alla fine.

```text
begin batch
  add
  scale
  matmul
end batch -> attesa e controllo errori
```

Il vantaggio cresce quando una rete esegue molte operazioni consecutive.

## Come lavorano i kernel paralleli

Le operazioni elementwise assegnano piccoli vettori a ogni thread. Riduzioni,
softmax e cross-entropy assegnano una riga a un gruppo di thread, che collaborano
e combinano i risultati parziali.

Scatter-add puo' avere piu' thread che aggiornano la stessa posizione. Usa
quindi un aggiornamento atomico: ogni somma viene applicata senza perdere le
altre.

## Matmul a tile

La moltiplicazione matriciale e' il calcolo piu' importante di un Transformer.
Leggere ripetutamente gli stessi valori dalla memoria principale sarebbe
costoso. I thread caricano piccoli blocchi, chiamati tile, in memoria veloce e li
riusano.

```text
matrice A -> tile A --+
                      +-> gruppo di thread -> blocco del risultato
matrice B -> tile B --+
```

Il backend dispone di tile 16 x 16, tile 32 x 32 e, sulle GPU recenti, matrici
SIMD 8 x 8. La selezione resta un dettaglio interno e non cambia il contratto
pubblico della matmul F32.

## Perche' il v1 usa soltanto FP32

FP32 usa 32 bit per valore ed e' il solo formato floating-point del contratto
v1. FP16 e BF16 possono ridurre memoria e traffico, ma richiedono cast, regole
di accumulo, loss scaling e nuovi test numerici.

Per evitare che un solo backend anticipi una semantica non richiesta, storage,
cast e matmul FP16/BF16 non sono implementati. Gli enum restano riservati per
una futura revisione comune di documenti, header e suite contrattuale.

## Come sappiamo se e' veloce

Il benchmark mostra due tempi:

- **end-to-end**: cio' che osserva davvero il programma;
- **GPU**: il solo intervallo eseguito dal dispositivo.

La differenza comprende preparazione, invio, sincronizzazione e controlli. Per
questo un kernel GPU veloce puo' perdere contro la CPU su tensori piccoli.

Il benchmark riporta anche throughput, tipo numerico, device e costo iniziale
delle pipeline. I numeri servono a confrontare modifiche sulla stessa macchina,
non a promettere la stessa velocita' su ogni Mac.

## Dove siamo davvero

Il backend gestisce memoria, batch e primitive F32 di base. Prima dei layer deve
ancora raggiungere la parita' col riferimento CPU per tutte le primitive di
training e superare la suite contrattuale comune su un device Apple reale.

Non usa precisione ridotta o fallback CPU. SiLU, RMSNorm, RoPE, attention,
backward e AdamW sono il prossimo lavoro concreto; fusion e funzionalita' future
restano fuori finche' il training corrente non le richiede.

I dettagli concreti e i comandi sono in
[docs/backend-metal.md](../docs/backend-metal.md).
