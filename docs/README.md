# Documentazione implementativa

La directory `docs/` contiene i contratti tecnici dei componenti reali del
progetto. Specifica formati, API, ownership, errori, invarianti e verifiche.

La [wiki](../wiki/README.md) introduce invece la teoria con un linguaggio piu'
semplice. Quando un argomento compare in entrambi i luoghi, conviene leggere
prima la pagina wiki e poi la specifica corrispondente.

## Ordine di lettura

1. [Toolchain](TOOLCHAIN.md)
2. [Corpus](corpus.md)
3. [Corpus multi-sorgente](corpus-multi-sorgente.md)
4. [Tokenizer](tokenizer.md)
5. [Dataset autoregressivo](dataset.md)
6. [Runtime tensoriale](runtime-tensoriale.md)
7. [Runtime v1: training minimo su Apple Silicon](runtime-v1-architecture.md)
8. [Modello Minimal: prima rete addestrabile](model-minimal.md)
9. [Italiano-Base-75M: primo decoder utilizzabile](italiano-base-75m.md)
10. [Checkpoint pubblici di Italiano Base 75M](italiano-base-75m-checkpoints.md)
11. [Italiano-Chat-75M: consolidamento e SFT](italiano-chat-75m.md)
12. [Serving e inferenza incrementale](serving-inference.md)
13. [Backend CPU](backend-cpu.md)
14. [Backend Metal](backend-metal.md)
15. [Backend CUDA](backend-cuda.md)
16. [Piano Metal ad alte prestazioni con MPS](metal-mps-development-plan.md)

Le specifiche descrivono anche componenti non ancora implementati. Ogni pagina
deve indicare chiaramente confini, stato atteso e criteri di completamento.
