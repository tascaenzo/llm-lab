# Pipeline RunPod CUDA

Questa pipeline usa una RTX 4090 singola (CUDA architecture `89`), conserva
dataset, checkpoint e log nel volume persistente `/workspace`, ed esegue sempre
la suite `runtime.cuda_backend` prima di modificare un checkpoint.

L'immagine e lo script di creazione impostano `LLM_LAB_BACKEND=cuda`. La
configurazione locale in `.env` e quella del deploy in `deploy/runpod/.env` sono
separate: la prima non viene letta dalla pipeline cloud. L'entrypoint rifiuta
qualsiasi backend diverso da CUDA, cosi' non puo' partire per errore un training
cloud sulla CPU.

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

Installa e configura `runpodctl`, copia `deploy/runpod/.env.example` in
`deploy/runpod/.env` e compila i valori cloud. Gli script caricano questo file
automaticamente, anche se vengono lanciati da un'altra directory; variabili gia'
esportate hanno precedenza. Quindi crea un Pod on-demand con
volume da 40 GB (gli input attuali occupano circa 14 GB prima di build e log):

```sh
cp deploy/runpod/.env.example deploy/runpod/.env
# Modifica deploy/runpod/.env; per il primo test imposta LLM_LAB_PROFILE_CUDA=1.
./deploy/runpod/setup.sh
./deploy/runpod/create_pod.sh
```

Per un file di configurazione cloud in un'altra posizione usa
`RUNPOD_ENV_FILE=/percorso/file`; `LLM_LAB_ENV_FILE` resta invece riservata ai
comandi locali.

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
`best.llmckpt`, metadati e log. I trasferimenti interrotti sono riprendibili:
le versioni moderne di rsync usano `--append-verify`, mentre l'rsync incluso in
macOS usa automaticamente la modalita' compatibile `--append`. Al termine crea
il marker di input completo.

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
