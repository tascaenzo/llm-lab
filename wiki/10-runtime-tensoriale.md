# Runtime tensoriale: il motore di calcolo

Il tokenizer trasforma il testo in ID e il dataset organizza questi ID in batch.
Prima di costruire la rete neurale serve un componente intermedio: il **runtime
tensoriale**.

Il runtime non decide cosa deve imparare il modello. Il suo compito e' conservare
i numeri e svolgere i calcoli richiesti nel modo corretto sull'hardware
disponibile.

```text
dataset -> modello -> runtime -> hardware
```

Il modello chiedera', per esempio, di moltiplicare due matrici. Il runtime
decidera' come eseguire quella richiesta sulla CPU o, in futuro, su una GPU.

## 1. Perche' serve uno strato intermedio

Senza un runtime, ogni layer dovrebbe contenere direttamente cicli e istruzioni
specifiche per un particolare processore:

```text
layer lineare -> cicli C per la CPU
attenzione    -> altri cicli C per la CPU
MLP           -> altri cicli C per la CPU
```

In questo modo il modello sarebbe difficile da leggere e dovrebbe essere
riscritto per usare Metal o CUDA.

Con il runtime, il modello descrive soltanto la matematica:

```text
layer lineare -> moltiplicazione + bias
```

Il runtime si occupa dell'esecuzione:

```text
moltiplicazione
    -> implementazione CPU
    -> implementazione Metal
    -> implementazione CUDA
```

## 2. Tensori: i contenitori dei numeri

Un tensore e' un insieme di numeri organizzato in una o piu' dimensioni.
Scalari, vettori e matrici sono tutti casi particolari di tensore.

| Esempio | Forma | Significato possibile |
|---|---|---|
| `2.31` | nessuna dimensione | loss |
| `[0.2, -0.5, 0.8]` | `[3]` | embedding di un token |
| tabella di righe e colonne | `[128, 256]` | pesi di un layer |
| blocco tridimensionale | `[8, 64, 128]` | batch, token, caratteristiche |

Il tensore conserva sia i numeri sia le informazioni necessarie per
interpretarli:

- forma;
- tipo numerico;
- posizione in memoria;
- dispositivo sul quale si trovano.

Un tensore non e' un layer e non e' un neurone. E' il contenitore usato da tutti
i componenti numerici.

## 3. Gestione della memoria

I tensori possono occupare molta memoria. Allocare e liberare continuamente
buffer durante ogni passaggio renderebbe il training lento e fragile.

Il gestore della memoria deve sapere come:

- riservare spazio;
- liberarlo quando non serve piu';
- riutilizzare buffer temporanei;
- copiare dati;
- trasferire dati tra CPU e GPU;
- impedire che una vista usi memoria gia' liberata.

La memoria non viene ottenuta nello stesso modo su ogni dispositivo:

```text
CPU    -> memoria del processo
Metal  -> buffer gestito da Metal
CUDA   -> memoria gestita da CUDA
```

Il modello non deve conoscere queste differenze.

## 4. Operazioni

Le operazioni sono il vocabolario matematico esposto dal runtime. Le prime
operazioni necessarie saranno:

- copia e azzeramento;
- somma e moltiplicazione elemento per elemento;
- moltiplicazione matriciale;
- somma, massimo e media di una serie di valori;
- selezione delle righe di un embedding;
- softmax e cross-entropy.

Il modello usera' queste operazioni per costruire calcoli piu' grandi. Per
esempio, un layer lineare esegue:

```text
output = input x pesi + bias
```

## 5. Backend: il coordinatore dell'hardware

Il backend rappresenta un ambiente di esecuzione. Conosce il dispositivo, la
sua memoria e le implementazioni disponibili.

Avremo inizialmente un solo backend:

```text
backend CPU
```

In futuro potranno essere aggiunti:

```text
backend Metal -> GPU Apple
backend CUDA  -> GPU NVIDIA
```

Il backend si occupa di:

- creare e distruggere il contesto del dispositivo;
- allocare la memoria corretta;
- scegliere il kernel adatto;
- avviare il calcolo;
- attendere il completamento quando necessario;
- trasformare gli errori hardware in errori comprensibili dal progetto.

## 6. Kernel: il calcolo concreto

Un kernel e' l'implementazione reale di un'operazione per un determinato
hardware.

Per la stessa moltiplicazione matriciale possono esistere piu' kernel:

```text
CPU    -> cicli C, SIMD oppure una libreria BLAS
Metal  -> programma eseguito dalla GPU Apple
CUDA   -> programma eseguito dalla GPU NVIDIA
```

Il risultato matematico deve essere lo stesso, entro le normali piccole
differenze dovute alla precisione numerica.

Il kernel non e' il driver. Il driver viene fornito dal sistema operativo o dal
produttore della GPU. Il nostro backend usa le API disponibili, come Metal o
CUDA, e queste comunicano con il driver.

## 7. Il percorso completo di un'operazione

Supponiamo che un layer debba calcolare `C = A x B`:

```text
modello
  |
  | richiede matmul(A, B)
  v
API del runtime
  |
  | controlla forme e tipi
  v
backend
  |
  | sceglie l'implementazione del dispositivo
  v
kernel
  |
  | esegue la moltiplicazione
  v
hardware
  |
  v
tensore C
```

Questa separazione permette di cambiare hardware senza riscrivere il modello.

## 8. Perche' iniziare dalla CPU

Il primo backend e' CPU. I kernel semplici e leggibili restano il riferimento
per verificare la matematica; accanto a essi verranno aggiunte implementazioni
parallele e ottimizzate.

Quando aggiungeremo Metal o CUDA, eseguiremo gli stessi test su ogni backend:

```text
risultato CPU circa uguale a risultato GPU
```

Ora che la versione corretta esiste, il prossimo incremento introduce
multithreading controllato. SIMD, librerie BLAS e kernel fusi verranno aggiunti
in seguito e soltanto dopo misure riproducibili.

## 9. Cosa non appartiene al runtime

Il runtime non contiene:

- l'architettura Transformer;
- i layer neurali;
- il significato degli embedding;
- il training loop;
- l'optimizer;
- la generazione del testo.

Questi componenti useranno il runtime, ma resteranno separati. In questo modo il
codice numerico puo' essere testato senza dover addestrare un modello.

## 10. Ordine di costruzione

Costruiremo il runtime in piccoli incrementi verificabili:

1. tipi comuni ed errori;
2. tensori e forma;
3. memoria CPU;
4. operazioni comuni;
5. backend CPU;
6. kernel CPU;
7. test matematici;
8. executor CPU e parallelizzazione;
9. memoria temporanea riutilizzabile;
10. backend accelerati futuri.

La specifica concreta delle strutture, delle responsabilita' e dei test e' in
[docs/runtime-tensoriale.md](../docs/runtime-tensoriale.md).

Il backend CPU di riferimento e' implementato insieme a tensori, memoria,
operazioni elementwise, riduzioni, matmul, gather/scatter, softmax e
cross-entropy. Prima di costruire backward e layer neurali, il backend viene
dotato di una struttura dedicata, un executor multithread e kernel paralleli. La
spiegazione continua in [Backend CPU](11-backend-cpu.md).
