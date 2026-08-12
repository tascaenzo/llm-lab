# Backend CPU

**Stato:** riferimento completo del contratto Runtime v1 F32/U32.
**Piattaforme:** macOS e Linux con pthread. Windows non e' supportato.

## Ruolo

Il backend CPU e' l'oracolo leggibile usato per validare contratti, gradienti e
future implementazioni Metal. Implementa soltanto le operazioni presenti nella
vtable runtime corrente; non contiene cast, precisione ridotta, KV cache o
funzioni speculative.

## Configurazione

```c
typedef struct llm_cpu_backend_config {
    size_t thread_count; /* 0 = CPU online */
} llm_cpu_backend_config;
```

Il totale include il thread chiamante. La funzione di creazione semplice usa
`thread_count=0`. Non esiste un'opzione di determinismo: l'ordine di riduzione
necessario alla riproducibilita' e' una proprieta' dell'implementazione.

## Executor

Il backend crea un pool pthread persistente. `llm_cpu_parallel_for` divide un
intervallo in chunk disgiunti, pubblica il job, usa anche il chiamante e attende
il completamento.

Se una callback invoca nuovamente `parallel_for` sullo stesso executor, la
chiamata interna viene eseguita inline. Un identificatore thread-local conserva
l'executor attivo e viene ripristinato al termine. Questa regola evita il
deadlock in cui tutti i worker attendono un job interno che nessuno puo'
eseguire.

Le chiamate annidate su executor diversi restano normali chiamate parallele.

## Memoria e dtype

L'allocazione usa `aligned_alloc` e `free`; non esistono rami Win32. Il
backend accetta soltanto F32 e U32. Gli enum F16, BF16 e F8 sono sempre
rifiutati.

Tutti i kernel richiedono tensori contigui gia' validati dalla facciata runtime.
I loop interni non allocano. Scatter-add viene partizionato in modo che ogni
task possieda colonne distinte, evitando atomiche e buffer temporanei.

## Operazioni implementate

- gestione tensori, reshape, read/write e copy;
- elementwise e accumulate;
- riduzioni sull'ultima dimensione;
- matmul F32 normale e con transpose logiche;
- gather/scatter-add;
- SiLU, RMSNorm e relativi backward;
- RoPE full-sequence e backward;
- attention GQA causale full-sequence e backward;
- softmax e cross-entropy;
- AdamW F32.

RoPE usa tabelle esatte `[S,D/2]`. Attention usa Q
`[B,S,Hq,D]` e K/V `[B,S,Hkv,D]` con la stessa S; offset e KV cache non
esistono nel backend.

## Riproducibilita' numerica

A parita' di input, thread count e build, la partizione e l'ordine delle
riduzioni sono stabili. Cambiare thread count puo' cambiare gli ultimi bit di
operazioni floating-point associative; la conformita' usa tolleranze specifiche
per operazione e non promette uguaglianza bit-a-bit fra configurazioni diverse.

Le opzioni non finite vengono respinte dalla facciata comune. Softmax,
cross-entropy e AdamW proteggono i casi numericamente non validi previsti dal
contratto.

## Test

La suite `backend_contract_suite.c` riceve un `llm_backend *` e non contiene
dipendenze CPU. Il driver CPU la esegue con 1 e 4 thread. Comprende:

- matmul transpose e regole di alias;
- gradient check SiLU, RMSNorm e attention;
- round-trip RoPE e tabelle di shape errata;
- causalita' attention e rifiuto di Q/K/V con S diverse;
- aggiornamento AdamW.

`test_cpu_backend.c` copre separatamente lifecycle, thread count, executor e
parallel-for annidato.

## Criterio di completamento

Il backend CPU e' pronto come riferimento Metal quando build Debug, intera
CTest e sanitizer passano, e quando `rg` non trova API fuori contratto nel
backend. Ottimizzazioni SIMD o nuove strategie di tiling sono ammesse solo se
non cambiano la vtable e continuano a superare la stessa suite.
