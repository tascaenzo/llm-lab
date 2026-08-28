# Valutazione di completamento — Italiano-Base-75M

`italiano-base-75m-prompts.jsonl` e' una suite versionata di 100 prompt italiani.
Ogni riga contiene un identificatore stabile e un prompt UTF-8.

Per ciascun checkpoint candidato, genera con configurazione fissa (`temperature`,
`top-k`, repetition penalty e seed), quindi registra per ogni ID il testo prodotto
e una valutazione manuale su italiano, aderenza al prompt, ripetizione e coerenza
breve. Il report deve indicare checksum del checkpoint, tokenizer, commit, comando
e parametri di generazione. Il test split non deve essere usato per scegliere
prompt, parametri di generazione o checkpoint.

La suite e' qualitativa: completa la validation quantitativa, non la sostituisce.

`italiano-chat-75m-prompts.jsonl` verifica invece il comportamento istruzionale:
aderenza, formato, ragionamento elementare, gestione dell'incertezza, richieste
di chiarimento e sicurezza. Ogni record dichiara categoria e criteri manuali.
La suite va eseguita con `llm-lab model chat`, mantenendo fissi checkpoint,
tokenizer, backend, temperatura, top-k, repetition penalty, seed e numero
massimo di token. Comando e output vanno registrati insieme alla revisione
manuale. I prompt di valutazione non devono essere copiati nel corpus SFT.
