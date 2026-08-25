# Pipeline RunPod CUDA

Questa pipeline usa una RTX 4090 singola (CUDA architecture `89`), conserva
dataset, checkpoint e log nel volume persistente `/workspace`, ed esegue sempre
la suite `runtime.cuda_backend` prima di modificare un checkpoint.

L'immagine rifiuta qualsiasi backend diverso da CUDA, cosi' non puo' partire per
errore un training cloud sulla CPU. Esiste un solo file di configurazione locale
ignorato da Git: `deploy/runpod/.env`. Contiene connessione, identita', Pod e
training; gli script inoltrano al container soltanto la lista esplicita di
chiavi `LLM_LAB_*` prevista dalla pipeline.

Il checkpoint Metal e il dataset sono portabili: la pipeline riprende
il checkpoint sorgente con il backend CUDA configurato nell'ambiente, senza conversione. Batch e
gradient accumulation possono essere ridistribuiti al resume purche' il loro prodotto resti
identico, quindi la dimensione effettiva dell'update e lo stato del sampler non cambiano.

## 1. Pubblicare l'immagine

La workflow manuale `.github/workflows/runpod-image.yml` costruisce e pubblica
l'immagine in GHCR. Dalla pagina Actions avvia `Build RunPod CUDA image` e
seleziona il branch `feat/cuda-backend`; il tag da usare su RunPod e'
`ghcr.io/tascaenzo/llm-lab-cuda:latest`.

L'immagine non contiene dati, checkpoint o segreti: `.dockerignore` li esclude.

## 2. Creare il Pod

Installa e configura `runpodctl`, poi crea l'unico file locale. Gli script
caricano `.env` automaticamente anche se vengono lanciati da un'altra
directory; variabili gia' esportate hanno precedenza. Configura qui account,
SSH, training, checkpoint, validation, matematica CUDA e profiling. Quindi crea un Pod
on-demand con volume da 40 GB (gli input attuali occupano circa 14 GB prima di
build e log):

```sh
cp deploy/runpod/.env.example deploy/runpod/.env
# Modifica deploy/runpod/.env.
./deploy/runpod/setup.sh
./deploy/runpod/create_pod.sh
```

Per usare un file diverso imposta `RUNPOD_ENV_FILE=/percorso/.env`;
`LLM_LAB_ENV_FILE` resta invece riservata ai comandi locali.

La chiave non viene scritta nel repository. Cambia `RUNPOD_GPU_ID` solo con una
GPU la cui CUDA architecture sia supportata dall'immagine; il default RTX 4090
usa architecture `89`. Prima di creare il Pod, aggiungi la tua chiave pubblica
SSH all'account RunPod: `setup.sh` esegue `runpodctl doctor`, che valida la API
key e configura questo passaggio. Il container espone TCP 22 e usa quella
chiave per i trasferimenti `rsync`.

## 3. Caricare lo stato locale una sola volta

Dopo la creazione, recupera host e porta SSH con `runpodctl ssh info POD_ID` e
salvali in `deploy/runpod/.env`. Il target SSH usa lo stesso formato di `ssh`,
per esempio. `setup.sh` crea gia' la chiave privata locale e gli script la
usano automaticamente; imposta `RUNPOD_SSH_IDENTITY_FILE` solo se hai usato una
chiave diversa.

```sh
# In deploy/runpod/.env:
# RUNPOD_SSH_HOST='root@HOST'
# RUNPOD_SSH_PORT='PORTA'
bash deploy/runpod/sync_to_pod.sh
```

Lo script trasferisce `italiano-v3` (train e validation) e `latest.llmckpt`,
`best.llmckpt`, metadati e log. I grandi file dataset, che sono immutabili,
riprendono i trasferimenti interrotti con `--append-verify` oppure con la
modalita' compatibile `--append` su macOS. Checkpoint e log usano invece una
sincronizzazione normale, cosi' un checkpoint nuovo ma della stessa dimensione
non puo' essere scambiato per quello vecchio. Al termine crea il marker di input
completo.

## 4. Avvio e ripresa

Al primo avvio il container resta in attesa del marker di upload: e' normale e
serve a permettere la copia sicura sul volume. Dopo `sync_to_pod.sh`, arresta e
riavvia il Pod con `runpodctl pod stop POD_ID` e `runpodctl pod start POD_ID`.
All'avvio successivo esegue la parita' CUDA e un vero step di training su un
checkpoint temporaneo in modalita' F32 rigorosa. Verifica inoltre che l'hash del checkpoint
sorgente non sia cambiato, poi lancia il resume reale verso un file candidato distinto. Il
preflight viene ripetuto solo quando
cambiano i binari dell'immagine. A ogni checkpoint il volume contiene uno stato
riavviabile.

Per fermare i costi, arresta il Pod (non eliminarlo): il volume persiste. Per
riprendere, riavvialo con le stesse variabili; il comando riparte dal checkpoint
piu' recente. Le variabili configurate sul Pod sono descritte nella documentazione
RunPod, e i file fuori da `/workspace` non sono persistenti.

Al termine di una sessione riuscita il container resta inattivo per evitare che
RunPod riavvii automaticamente altri step. Il Pod resta comunque a pagamento
finche' e' in esecuzione: scarica i risultati e fermalo esplicitamente.

La configurazione predefinita esegue 19.500 step, circa un'ora sulla RTX 4090
misurata durante il primo test, con checkpoint e validazione ogni 5.000 step.

## Aggiornare un Pod esistente

Inserisci il suo ID in `RUNPOD_POD_ID` nel file `.env`. A Pod fermo, modifica
immagine o configurazione e usa un solo comando:

```sh
./deploy/runpod/update_pod.sh
runpodctl pod start POD_ID
```

`update_pod.sh` usa la stessa configurazione di `create_pod.sh`; non occorrono
JSON incollati a mano nella shell. Non eseguirlo durante il training: la
modifica del Pod ricrea il container, mentre `/workspace` resta persistente.

### Checkpoint sorgente, candidato e promozione

`LLM_LAB_RESUME_CHECKPOINT` e' sempre di sola lettura durante il run. Il risultato viene scritto
in `LLM_LAB_OUTPUT_CHECKPOINT`; il best e il log hanno percorsi separati. I default usano
`latest.llmckpt` come sorgente e `candidate-fast.llmckpt` come output, quindi una regressione non
puo' cancellare i progressi esistenti. Dopo il canary e la validation, promuovi il candidato a
`latest.llmckpt` solo con una copia esplicita e conserva il vecchio latest finche' non hai
verificato il nuovo file.

Se il Pod viene fermato dopo il primo salvataggio, al riavvio la pipeline rileva il candidato e
riprende automaticamente da quello, non dal baseline. Se il candidato e' corrotto il loader
fallisce senza fallback silenzioso: `latest` resta disponibile per il rollback e il lavoro cloud
gia' salvato non viene ripetuto per errore.

Per una ripresa piu' efficiente, il checkpoint canonico `batch=4, accumulation=2` puo' per esempio
diventare `batch=8, accumulation=1` impostando le due variabili `LLM_LAB_RESUME_*`. Il loader
rifiuta combinazioni che cambiano il prodotto, cosi' non altera la semantica dell'update.

### Modalita' matematica CUDA

`LLM_LAB_CUDA_MATH` accetta `f32`, `tf32` e `bf16-compute`. Storage di parametri, gradienti,
optimizer e checkpoint rimangono F32 in tutti e tre i casi; cambia soltanto il percorso di calcolo
dei GEMM cuBLAS. `bf16-compute` usa i Tensor Core BF16 con accumulo e output F32 ed e' il primo
candidato da misurare sulla RTX 4090. `tf32` resta disponibile per confronto. `f32` e' il default
prudente e viene sempre forzato dal preflight.

`LLM_LAB_CUDA_NUMERICS=strict` controlla ogni output come prima. `step` elimina le scansioni delle
attivazioni intermedie, ma conserva i controlli obbligatori su loss, indici e parametri master
dopo AdamW. Usalo soltanto insieme al checkpoint candidato separato e dopo un canary valido.

Per misurare senza avviare il training usa `LLM_LAB_RUN_MODE=profile-only`. Per il canary usa
`train`, un output candidato, 200-1.000 step e la stessa validation del run F32. Confronta
token/s, loss e perplexity; promuovi la combinazione veloce solo se le metriche restano sane.
L'avvio stampa esplicitamente matematica e politica numerica effettive.

### Recuperare un volume senza GPU

`LLM_LAB_RUN_MODE=transfer-only` avvia soltanto SSH e resta inattivo: non chiama CUDA e non
avvia il trainer. Serve per montare un volume persistente su un Pod CPU e recuperare checkpoint
e log quando una GPU non e' disponibile. Imposta temporaneamente la variabile, aggiorna e avvia
il Pod, esegui `sync_from_pod.sh`, quindi fermalo. Ripristina `train` prima della prossima
sessione CUDA.

## Profilare prima di spendere

Imposta `LLM_LAB_PROFILE_CUDA=1` quando crei un Pod diagnostico. Dopo il
preflight, ma prima di cambiare il checkpoint, la pipeline misura in batch le
forme e il numero reale delle operazioni che compongono un update e salva il report in
`artifacts/benchmarks/runpod/cuda-profile-*.txt` nel volume. Controlla soprattutto
gli stadi e le operazioni in cima al report; poi scaricalo con
`sync_from_pod.sh`. Un marker legato alla versione dell'immagine impedisce di
ripeterlo ai riavvii. Per le normali sessioni lascia il valore a `0`.

Il risultato reale dello step usa-e-getta e il relativo JSONL sono conservati
come `artifacts/benchmarks/runpod/cuda-preflight-*`; il grande checkpoint
temporaneo viene invece eliminato dal disco del container appena verificato.

Ogni riga `event=train` di `training.jsonl` include inoltre i campi
`accelerator_*`: backend, memoria attiva e in cache, picco, kernel lanciati,
sincronizzazioni, riuso dei buffer e tempi GPU. I vecchi campi `metal_*`
restano presenti per compatibilita', anche quando il backend e' CUDA.

Se l'immagine GHCR e' privata, rendi pubblico il package `llm-lab-cuda` oppure
configura le credenziali del registry in RunPod prima di creare il Pod.

## 5. Riportare i risultati sul Mac

```sh
bash deploy/runpod/sync_from_pod.sh
```

Scarica `latest`, `best`, metriche e log. Non avviare piu' sessioni Metal sullo
stesso checkpoint dopo l'avvio CUDA: scegli un solo writer per `latest.llmckpt`.
