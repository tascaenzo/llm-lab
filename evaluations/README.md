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
