# Pipeline RunPod CUDA

Questa pipeline usa una RTX 4090 singola (CUDA architecture `89`), conserva
dataset, checkpoint e log nel volume persistente `/workspace`, ed esegue sempre
la suite `runtime.cuda_backend` prima di modificare un checkpoint.

Il checkpoint Metal e il dataset sono portabili: la pipeline riprende
`latest.llmckpt` con `--backend cuda`, senza conversione e senza modificare gli
iperparametri salvati nel checkpoint.

## 1. Pubblicare l'immagine

La workflow manuale `.github/workflows/runpod-image.yml` costruisce e pubblica
l'immagine in GHCR. Avviala dal branch `feat/cuda-backend`; il tag da usare su
RunPod e' `ghcr.io/tascaenzo/llm-lab-cuda:latest`.

L'immagine non contiene dati, checkpoint o segreti: `.dockerignore` li esclude.

## 2. Creare il Pod

Installa e configura `runpodctl`, quindi crea un Pod on-demand con volume da
40 GB (gli input attuali occupano circa 14 GB prima di build e log):

```sh
export RUNPOD_API_KEY='...'
export LLM_LAB_TRAIN_STEPS=1000
bash deploy/runpod/create_pod.sh
```

La chiave non viene scritta nel repository. Cambia `RUNPOD_GPU_ID` solo con una
GPU la cui CUDA architecture sia supportata dall'immagine; il default RTX 4090
usa architecture `89`. Prima di creare il Pod, aggiungi la tua chiave pubblica
SSH all'account RunPod (`runpodctl doctor` lo configura): il container espone
TCP 22 e usa quella chiave per i trasferimenti `rsync`.

## 3. Caricare lo stato locale una sola volta

Dopo la creazione, recupera host e porta SSH con `runpodctl pod get POD_ID`.
Poi usa il target SSH nello stesso formato di `ssh`, per esempio:

```sh
export RUNPOD_SSH_HOST='root@HOST'
export RUNPOD_SSH_PORT='PORTA'
bash deploy/runpod/sync_to_pod.sh
```

Lo script trasferisce `italiano-v3` (train e validation) e `latest.llmckpt`,
`best.llmckpt`, metadati e log. `--partial --append-verify` consente di
riprendere upload interrotti. Al termine crea il marker di input completo.

## 4. Avvio e ripresa

Al primo avvio il container resta in attesa del marker di upload: e' normale e
serve a permettere la copia sicura sul volume. Dopo `sync_to_pod.sh`, arresta e
riavvia il Pod con `runpodctl pod stop POD_ID` e `runpodctl pod start POD_ID`.
All'avvio successivo esegue la parita' CUDA su GPU reale, poi lancia il resume
da `latest.llmckpt`. A ogni checkpoint il volume contiene uno stato riavviabile.

Per fermare i costi, arresta il Pod (non eliminarlo): il volume persiste. Per
riprendere, riavvialo con le stesse variabili; il comando riparte dal checkpoint
piu' recente. Le variabili configurate sul Pod sono descritte nella documentazione
RunPod, e i file fuori da `/workspace` non sono persistenti.

Se l'immagine GHCR e' privata, rendi pubblico il package `llm-lab-cuda` oppure
configura le credenziali del registry in RunPod prima di creare il Pod.

## 5. Riportare i risultati sul Mac

```sh
export RUNPOD_SSH_HOST='root@HOST'
export RUNPOD_SSH_PORT='PORTA'
bash deploy/runpod/sync_from_pod.sh
```

Scarica `latest`, `best`, metriche e log. Non avviare piu' sessioni Metal sullo
stesso checkpoint dopo l'avvio CUDA: scegli un solo writer per `latest.llmckpt`.
