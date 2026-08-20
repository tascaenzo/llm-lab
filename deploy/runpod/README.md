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
`latest.llmckpt` con il backend CUDA configurato nell'ambiente, senza conversione e senza modificare gli
iperparametri salvati nel checkpoint.

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
SSH, training, checkpoint, validation, TF32 e profiling. Quindi crea un Pod
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
checkpoint temporaneo. Verifica inoltre che l'hash di `latest.llmckpt` non sia
cambiato, poi lancia il resume reale. Il preflight viene ripetuto solo quando
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

### Test TF32 prima del run lungo

`LLM_LAB_CUDA_TF32=1` nel file `.env` abilita i Tensor Core per i GEMM cuBLAS. Storage dei
parametri, gradienti, optimizer e checkpoint rimangono F32, ma i prodotti
matrice-matrice non sono bit-exact rispetto al percorso F32 rigoroso. Il valore
predefinito e' `0`.

Per provarlo senza rischiare il run lungo, prima scarica o copia il checkpoint
latest corrente, poi aggiorna temporaneamente il Pod con 1.000 step, TF32 e la
stessa validation. Confronta step/s e validation loss/perplexity con l'ultimo
run F32. Mantieni TF32 per le sessioni lunghe solo se non introduce anomalie
nelle metriche. L'avvio stampa esplicitamente la modalita' di matematica scelta.

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
