# Documentazione implementativa

La directory `docs/` contiene i contratti tecnici dei componenti reali del
progetto. Specifica formati, API, ownership, errori, invarianti e verifiche.

La [wiki](../wiki/README.md) introduce invece la teoria con un linguaggio piu'
semplice. Quando un argomento compare in entrambi i luoghi, conviene leggere
prima la pagina wiki e poi la specifica corrispondente.

## Ordine di lettura

1. [Toolchain](TOOLCHAIN.md)
2. [Corpus](corpus.md)
3. [Tokenizer](tokenizer.md)
4. [Dataset autoregressivo](dataset.md)
5. [Runtime tensoriale](runtime-tensoriale.md)
6. [Backend CPU](backend-cpu.md)
7. [Backend Metal](backend-metal.md)

Le specifiche descrivono anche componenti non ancora implementati. Ogni pagina
deve indicare chiaramente confini, stato atteso e criteri di completamento.
