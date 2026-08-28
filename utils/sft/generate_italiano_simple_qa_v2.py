#!/usr/bin/env python3
"""Generate a broad, balanced and controlled Italian QA/SFT corpus.

Version 2 retains the reviewed simple-QA v1 examples and adds balanced
practical, writing, clarification, uncertainty and stable knowledge examples.
All variants of one semantic group remain in one native split.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import generate_italiano_simple_qa_v1 as v1


SOURCE = "local-curated:italiano-simple-qa-v2"
SCRIPT = "utils/sft/generate_italiano_simple_qa_v2.py"


SCIENCE = [
    ("rotazione-terra", "Quanto dura approssimativamente una rotazione completa della Terra?", "Una rotazione completa della Terra dura circa 24 ore."),
    ("rivoluzione-terra", "Quanto dura approssimativamente un'orbita della Terra intorno al Sole?", "Un'orbita completa dura circa 365 giorni e un quarto."),
    ("stagioni", "Perché sulla Terra esistono le stagioni?", "Le stagioni dipendono soprattutto dall'inclinazione dell'asse terrestre durante l'orbita intorno al Sole."),
    ("mare-salato", "Perché l'acqua del mare è salata?", "Perché contiene sali minerali disciolti trasportati principalmente da rocce, fiumi e attività geologica."),
    ("evaporazione", "Che cos'è l'evaporazione?", "È il passaggio di una sostanza dallo stato liquido a quello aeriforme dalla sua superficie."),
    ("condensazione", "Che cos'è la condensazione?", "È il passaggio di una sostanza dallo stato aeriforme a quello liquido."),
    ("ciclo-acqua", "Che cos'è il ciclo dell'acqua?", "È il movimento continuo dell'acqua tra superficie terrestre e atmosfera tramite evaporazione, condensazione e precipitazioni."),
    ("suono", "Come si produce un suono?", "Un suono nasce da una vibrazione che si propaga attraverso un mezzo, come aria, acqua o solidi."),
    ("conduttore", "Che cos'è un conduttore elettrico?", "È un materiale che permette alle cariche elettriche di muoversi con relativa facilità."),
    ("isolante", "Che cos'è un isolante elettrico?", "È un materiale che ostacola il movimento delle cariche elettriche."),
    ("magnete", "Che cos'è un magnete?", "È un oggetto che produce un campo magnetico e può attrarre alcuni materiali, come il ferro."),
    ("cellula", "Che cos'è una cellula?", "La cellula è l'unità strutturale e funzionale di base degli organismi viventi."),
    ("dna", "A che cosa serve il DNA?", "Il DNA conserva gran parte delle informazioni genetiche necessarie allo sviluppo e al funzionamento degli organismi."),
    ("batteri", "Che cosa sono i batteri?", "I batteri sono organismi microscopici unicellulari; molti sono innocui o utili, mentre alcuni causano malattie."),
    ("erbivoro", "Che cos'è un animale erbivoro?", "È un animale che si nutre principalmente di vegetali."),
    ("carnivoro", "Che cos'è un animale carnivoro?", "È un animale che si nutre principalmente di altri animali."),
    ("onnivoro", "Che cos'è un animale onnivoro?", "È un animale che può nutrirsi sia di alimenti vegetali sia animali."),
    ("catena-alimentare", "Che cos'è una catena alimentare?", "È una rappresentazione del passaggio di materia ed energia tra organismi che si nutrono l'uno dell'altro."),
    ("atmosfera", "Che cos'è l'atmosfera terrestre?", "È l'involucro di gas che circonda la Terra."),
    ("ozono", "Quale funzione svolge lo strato di ozono?", "Assorbe gran parte della radiazione ultravioletta proveniente dal Sole."),
    ("fossile", "Che cos'è un fossile?", "È un resto o una traccia di un organismo del passato conservata nelle rocce o in altri materiali."),
    ("vulcano", "Che cos'è un vulcano?", "È una struttura geologica attraverso cui magma, gas e materiali possono raggiungere la superficie."),
    ("terremoto", "Che cos'è un terremoto?", "È una vibrazione del terreno causata dal rilascio improvviso di energia nella crosta terrestre."),
    ("erosione", "Che cos'è l'erosione?", "È il consumo e il trasporto di suolo o roccia dovuto ad acqua, vento, ghiaccio o gravità."),
    ("densita", "Che cos'è la densità?", "La densità indica quanta massa è contenuta in un determinato volume."),
    ("massa-peso", "Qual è la differenza tra massa e peso?", "La massa misura la quantità di materia; il peso è la forza esercitata dalla gravità su quella massa."),
    ("riflessione-luce", "Che cos'è la riflessione della luce?", "È il fenomeno per cui la luce cambia direzione quando incontra una superficie e torna nel mezzo di provenienza."),
    ("arcobaleno", "Come si forma un arcobaleno?", "Si forma quando la luce solare viene rifratta, riflessa e separata nei suoi colori dalle gocce d'acqua."),
    ("satellite", "Che cos'è un satellite naturale?", "È un corpo celeste che orbita naturalmente intorno a un pianeta o a un altro corpo maggiore."),
    ("galassia", "Che cos'è una galassia?", "È un grande insieme di stelle, gas, polveri e materia oscura legati dalla gravità."),
]

HUMANITIES = [
    ("garibaldi", "Chi era Giuseppe Garibaldi?", "Giuseppe Garibaldi fu un generale e patriota, protagonista del Risorgimento italiano."),
    ("cavour", "Chi era Camillo Benso, conte di Cavour?", "Fu uno statista piemontese e una figura centrale nel processo di unificazione italiana."),
    ("mazzini", "Chi era Giuseppe Mazzini?", "Fu un patriota e pensatore politico che sostenne l'unità e la repubblica italiana."),
    ("prima-guerra", "In quali anni si svolse la Prima guerra mondiale?", "La Prima guerra mondiale si svolse dal 1914 al 1918."),
    ("seconda-guerra", "In quali anni si svolse la Seconda guerra mondiale?", "La Seconda guerra mondiale si svolse dal 1939 al 1945."),
    ("repubblica-italiana", "In quale anno nacque la Repubblica Italiana?", "La Repubblica Italiana nacque nel 1946, dopo il referendum istituzionale."),
    ("roma-capitale", "In quale anno Roma divenne capitale del Regno d'Italia?", "Roma divenne capitale del Regno d'Italia nel 1871."),
    ("impero-romano", "Chi fu il primo imperatore romano?", "Augusto è considerato il primo imperatore romano."),
    ("magna-carta", "In quale paese fu concessa la Magna Carta del 1215?", "La Magna Carta fu concessa in Inghilterra."),
    ("rivoluzione-francese", "In quale anno iniziò la Rivoluzione francese?", "La Rivoluzione francese iniziò nel 1789."),
    ("leopardi", "Chi ha scritto L'infinito?", "L'infinito è una poesia di Giacomo Leopardi."),
    ("verga", "Chi ha scritto I Malavoglia?", "I Malavoglia è un romanzo di Giovanni Verga."),
    ("pirandello", "Chi ha scritto Sei personaggi in cerca d'autore?", "L'opera è stata scritta da Luigi Pirandello."),
    ("calvino", "Chi ha scritto Il barone rampante?", "Il barone rampante è un romanzo di Italo Calvino."),
    ("eco", "Chi ha scritto Il nome della rosa?", "Il nome della rosa è un romanzo di Umberto Eco."),
    ("shakespeare", "Chi ha scritto Romeo e Giulietta?", "Romeo e Giulietta è una tragedia di William Shakespeare."),
    ("odissea", "A chi viene tradizionalmente attribuita l'Odissea?", "L'Odissea viene tradizionalmente attribuita a Omero."),
    ("don-chisciotte", "Chi ha scritto Don Chisciotte della Mancia?", "Il romanzo è stato scritto da Miguel de Cervantes."),
    ("urlo", "Chi ha dipinto L'urlo?", "L'urlo è un'opera di Edvard Munch."),
    ("persistenza-memoria", "Chi ha dipinto La persistenza della memoria?", "La persistenza della memoria è un dipinto di Salvador Dalí."),
    ("cappella-sistina", "Chi dipinse gran parte della volta della Cappella Sistina?", "Gran parte della volta fu dipinta da Michelangelo Buonarroti."),
    ("quattro-stagioni", "Chi ha composto Le quattro stagioni?", "Le quattro stagioni furono composte da Antonio Vivaldi."),
    ("nona-beethoven", "Chi ha composto la Nona sinfonia?", "La Nona sinfonia fu composta da Ludwig van Beethoven."),
    ("aida", "Chi ha composto l'opera Aida?", "Aida fu composta da Giuseppe Verdi."),
    ("flauto-magico", "Chi ha composto Il flauto magico?", "Il flauto magico fu composto da Wolfgang Amadeus Mozart."),
]

INFORMATICS = [
    ("gpu", "GPU", "Una GPU è un processore specializzato nell'esecuzione parallela, usato soprattutto per grafica e calcoli intensivi."),
    ("url", "URL", "Un URL è l'indirizzo che identifica una risorsa sul web."),
    ("http", "HTTP", "HTTP è un protocollo usato per trasferire risorse tra client e server sul web."),
    ("https", "HTTPS", "HTTPS è HTTP protetto tramite cifratura e autenticazione della connessione."),
    ("ip", "indirizzo IP", "Un indirizzo IP identifica un dispositivo o un'interfaccia all'interno di una rete basata su IP."),
    ("router", "router", "Un router instrada i dati tra reti diverse."),
    ("modem", "modem", "Un modem collega una rete al servizio fornito dall'operatore convertendo i segnali necessari alla trasmissione."),
    ("server", "server", "Un server fornisce dati o servizi ad altri dispositivi chiamati client."),
    ("client", "client", "Un client richiede dati o servizi a un server."),
    ("dns", "DNS", "Il DNS traduce nomi leggibili, come quelli dei siti, negli indirizzi usati dalla rete."),
    ("cache", "cache", "Una cache conserva temporaneamente dati usati spesso per renderne più rapido l'accesso."),
    ("cookie", "cookie del browser", "Un cookie è un piccolo dato salvato dal browser per mantenere preferenze o informazioni di sessione."),
    ("codice-sorgente", "codice sorgente", "Il codice sorgente è il testo scritto dai programmatori per definire il comportamento di un programma."),
    ("compilatore", "compilatore", "Un compilatore traduce codice sorgente in una forma eseguibile o intermedia."),
    ("bug", "bug informatico", "Un bug è un errore nel software che produce un comportamento inatteso o scorretto."),
    ("open-source", "software open source", "È software il cui codice sorgente è disponibile secondo i termini della relativa licenza."),
    ("cifratura", "cifratura", "La cifratura trasforma i dati affinché siano leggibili solo con le informazioni necessarie a decifrarli."),
    ("phishing", "phishing", "Il phishing è un tentativo di ingannare una persona per ottenere dati o accessi riservati."),
    ("malware", "malware", "Un malware è software progettato per danneggiare, spiare o compromettere sistemi e dati."),
    ("firewall", "firewall", "Un firewall controlla il traffico di rete applicando regole di sicurezza."),
    ("aggiornamento", "aggiornamento software", "Un aggiornamento può correggere errori, migliorare sicurezza e aggiungere funzionalità."),
    ("pdf", "file PDF", "Un PDF è un formato di documento progettato per mantenere un aspetto coerente su dispositivi diversi."),
    ("foglio-calcolo", "foglio di calcolo", "Un foglio di calcolo organizza dati in celle e permette formule, calcoli e grafici."),
    ("videoscrittura", "programma di videoscrittura", "È un'applicazione usata per creare, modificare e formattare documenti di testo."),
    ("motore-ricerca", "motore di ricerca", "Un motore di ricerca aiuta a trovare pagine e risorse indicizzate sul web."),
    ("collegamento", "collegamento ipertestuale", "Un collegamento ipertestuale porta a un'altra risorsa o posizione quando viene attivato."),
    ("qr", "codice QR", "Un codice QR rappresenta dati in una matrice grafica leggibile da dispositivi compatibili."),
    ("usb", "porta USB", "Una porta USB permette di collegare dispositivi e può trasferire dati ed energia."),
    ("compressione", "compressione dei dati", "La compressione riduce lo spazio necessario per rappresentare i dati."),
    ("formato-file", "formato di file", "Un formato di file definisce come i dati sono organizzati e interpretati."),
]

EVERYDAY = [
    ("calendario", "calendario", "Un calendario organizza giorni, settimane e mesi e aiuta a registrare eventi e scadenze."),
    ("agenda", "agenda", "Un'agenda serve ad annotare appuntamenti, attività e promemoria."),
    ("promemoria", "promemoria", "Un promemoria segnala un'attività o un evento da ricordare."),
    ("scadenza", "scadenza", "Una scadenza è il termine entro cui un'attività deve essere completata."),
    ("checklist", "lista di controllo", "Una lista di controllo raccoglie elementi da verificare o attività da completare."),
    ("etichetta", "etichetta", "Un'etichetta identifica un oggetto o fornisce informazioni sul suo contenuto."),
    ("ricetta", "ricetta di cucina", "Una ricetta elenca ingredienti e passaggi necessari per preparare un piatto."),
    ("dispensa", "dispensa", "Una dispensa è lo spazio in cui si conservano alimenti e scorte che non richiedono frigorifero."),
    ("compost", "compost", "Il compost è materiale ottenuto dalla decomposizione controllata di residui organici."),
    ("riciclo", "riciclo", "Il riciclo trasforma materiali di scarto in risorse utilizzabili per nuovi prodotti."),
    ("termometro-casa", "termometro domestico", "Un termometro domestico misura la temperatura dell'ambiente o di ciò per cui è progettato."),
    ("termostato", "termostato", "Un termostato controlla un impianto per mantenere una temperatura impostata."),
    ("cronometro", "cronometro", "Un cronometro misura la durata di un intervallo di tempo."),
    ("bussola", "bussola", "Una bussola indica la direzione rispetto al campo magnetico terrestre."),
    ("mappa", "mappa", "Una mappa rappresenta in modo semplificato un territorio o uno spazio."),
    ("trasporto-pubblico", "trasporto pubblico", "Il trasporto pubblico comprende servizi condivisi come autobus, tram, metro e treni."),
    ("appuntamento", "appuntamento", "Un appuntamento è un incontro previsto in un luogo e a un orario concordati."),
    ("inventario", "inventario", "Un inventario è un elenco organizzato dei beni o prodotti disponibili."),
    ("archivio", "archivio", "Un archivio conserva documenti o dati organizzandoli per facilitarne la ricerca."),
    ("indice", "indice di un libro", "L'indice elenca parti o argomenti di un libro e indica dove trovarli."),
    ("dizionario", "dizionario", "Un dizionario raccoglie parole e ne descrive significati, usi o traduzioni."),
    ("enciclopedia", "enciclopedia", "Un'enciclopedia raccoglie informazioni organizzate su numerosi argomenti."),
    ("bibliografia", "bibliografia", "Una bibliografia elenca le fonti utilizzate o consigliate per un lavoro."),
    ("bozza", "bozza", "Una bozza è una versione provvisoria destinata a essere rivista."),
    ("revisione", "revisione di un testo", "La revisione controlla e migliora contenuto, chiarezza e correttezza di un testo."),
    ("riassunto", "riassunto", "Un riassunto presenta le informazioni principali di un testo in forma più breve."),
    ("titolo", "titolo", "Un titolo identifica un contenuto e ne anticipa l'argomento principale."),
    ("paragrafo", "paragrafo", "Un paragrafo riunisce frasi dedicate a uno stesso punto o argomento."),
    ("priorita", "priorità", "Una priorità è un'attività o un obiettivo a cui viene data precedenza."),
    ("routine", "routine", "Una routine è una sequenza di azioni ripetuta con regolarità."),
]

PRACTICAL = [
    ("studio-giorno", "Come organizzo una giornata di studio?", "Scegli due obiettivi realistici, assegna loro intervalli precisi e termina con un breve ripasso."),
    ("studio-settimana", "Come organizzo una settimana di studio?", "Distribuisci gli argomenti nei giorni disponibili, alterna studio e ripasso e lascia tempo per recuperare eventuali ritardi."),
    ("esame-orale", "Come mi preparo per un esame orale?", "Crea una scaletta degli argomenti, esercitati a voce e annota i punti che non riesci ancora a spiegare chiaramente."),
    ("flashcard", "Come creo flashcard utili?", "Scrivi una domanda precisa sul fronte e una risposta breve sul retro, dedicando ogni carta a un solo concetto."),
    ("appunti", "Come posso prendere appunti più chiari?", "Annota concetti e parole chiave, usa titoli riconoscibili e lascia spazio per aggiungere esempi o chiarimenti."),
    ("email-oggetto", "Come scelgo l'oggetto di un'email professionale?", "Usa poche parole che descrivano con precisione richiesta e contesto, per esempio «Richiesta informazioni ordine 125»."),
    ("email-allegato", "Come ricordo a qualcuno che manca un allegato?", "Puoi scrivere: «Grazie per il messaggio. Potresti inviarmi anche l'allegato citato, per favore?»."),
    ("email-risposta", "Come sollecito gentilmente una risposta?", "Puoi scrivere: «Buongiorno, riprendo il messaggio precedente per sapere se ci sono aggiornamenti. Grazie»."),
    ("riunione", "Come preparo una riunione breve?", "Definisci l'obiettivo, limita i punti all'ordine del giorno e assegna un tempo a ciascun punto."),
    ("verbale", "Come riassumo una riunione?", "Registra decisioni, attività da svolgere, responsabili e relative scadenze."),
    ("presentazione", "Come preparo una presentazione di cinque minuti?", "Definisci un messaggio centrale, sviluppa due o tre punti e chiudi con un riepilogo."),
    ("slide", "Come evito slide troppo affollate?", "Mantieni un'idea principale per slide, riduci il testo e usa solo elementi visivi che aiutano davvero la spiegazione."),
    ("file-progetto", "Come organizzo i file di un progetto?", "Usa una cartella principale e sottocartelle chiare per documenti, risorse, bozze e versioni finali."),
    ("nomi-file", "Come scelgo nomi di file comprensibili?", "Usa parole descrittive e, quando serve, aggiungi data o numero di versione in un formato coerente."),
    ("desktop", "Come riordino il desktop del computer?", "Sposta i file nelle cartelle corrette, elimina collegamenti inutili e lascia sul desktop solo ciò che usi spesso."),
    ("backup-foto", "Come organizzo il backup delle foto?", "Conserva almeno una seconda copia, automatizza il salvataggio quando possibile e verifica periodicamente alcuni file."),
    ("password", "Come gestisco molte password?", "Usa password uniche e lunghe, affidati a un gestore di password e attiva l'autenticazione a due fattori."),
    ("notifiche", "Come riduco le notifiche che mi distraggono?", "Mantieni quelle importanti e disattiva avvisi promozionali, duplicati o non urgenti."),
    ("scrivania", "Come riordino una scrivania in dieci minuti?", "Elimina ciò che non serve, raggruppa gli oggetti simili e lascia a portata di mano solo quelli usati spesso."),
    ("dispensa", "Come organizzo una piccola dispensa?", "Raggruppa gli alimenti per tipo, metti davanti quelli con scadenza vicina e rendi visibili le confezioni aperte."),
    ("frigorifero", "Come riordino il frigorifero?", "Controlla le scadenze, pulisci un ripiano alla volta e sistema gli alimenti nelle zone adatte."),
    ("lista-spesa", "Come preparo una lista della spesa efficace?", "Controlla ciò che hai già e dividi la lista per reparti, così eviti acquisti doppi e perdi meno tempo."),
    ("pasto", "Come preparo un pasto semplice quando ho poco tempo?", "Scegli una base, una fonte proteica e una verdura, usando pochi ingredienti che richiedano cotture compatibili."),
    ("valigia", "Come preparo una valigia leggera?", "Controlla meteo e attività, scegli capi combinabili e porta solo quantità adatte alla durata del viaggio."),
    ("appuntamento", "Come mi preparo per un appuntamento importante?", "Raccogli documenti e domande in anticipo e verifica orario, indirizzo e tempo necessario per arrivare."),
    ("budget-tempo", "Come pianifico una giornata piena di impegni?", "Individua le priorità, stima tempi realistici e lascia piccoli margini tra un'attività e l'altra."),
    ("progetto", "Come avvio un piccolo progetto personale?", "Definisci un risultato concreto, scegli il primo passo realizzabile e stabilisci quando verificarne l'avanzamento."),
    ("abitudine", "Come costruisco una nuova abitudine?", "Inizia con un'azione molto piccola, collegala a un momento stabile della giornata e registra la continuità."),
    ("lettura", "Come posso leggere con maggiore attenzione?", "Prima osserva struttura e titoli, poi annota domande e al termine riassumi a memoria le idee principali."),
    ("riassunto", "Come scrivo un riassunto chiaro?", "Individua le idee principali, elimina dettagli secondari e riscrivi i concetti con parole tue."),
    ("testo-lungo", "Come rendo più leggibile un testo lungo?", "Metti l'informazione principale all'inizio, dividi il testo in paragrafi e usa elenchi solo quando aiutano."),
    ("istruzioni", "Come scrivo istruzioni facili da seguire?", "Ordina i passaggi, descrivi una sola azione per punto e indica il risultato da ottenere."),
    ("feedback", "Come chiedo un feedback utile?", "Indica su quale parte vuoi un parere e formula domande specifiche invece di chiedere un giudizio generico."),
    ("errore", "Come comunico di aver commesso un errore?", "Riconosci l'errore, descrivi l'effetto concreto e spiega come intendi correggerlo."),
    ("ritardo", "Come avviso che arriverò in ritardo?", "Comunica subito il ritardo, fornisci una stima realistica e scusati in modo breve."),
    ("rifiuto", "Come rifiuto un invito con gentilezza?", "Ringrazia per l'invito, comunica chiaramente che non puoi partecipare e mantieni il messaggio breve."),
    ("priorita", "Come scelgo tra molte attività urgenti?", "Valuta scadenza e conseguenze, completa prima ciò che blocca altre persone e rimanda il resto in modo esplicito."),
    ("decisione", "Come confronto due alternative?", "Definisci i criteri importanti, assegna loro una priorità e confronta le alternative usando gli stessi criteri."),
    ("problema", "Come affronto un problema complesso?", "Descrivilo con precisione, dividilo in parti più piccole e verifica una soluzione alla volta."),
    ("obiettivo", "Come trasformo un'idea in un obiettivo concreto?", "Specifica il risultato osservabile, una scadenza realistica e il primo passo da completare."),
    ("concentrazione", "Come posso concentrarmi per un breve periodo?", "Scegli un compito preciso, allontana le distrazioni e imposta un intervallo breve senza interruzioni."),
    ("pausa", "Come uso bene una pausa dallo studio?", "Alzati, muoviti e riposa gli occhi, evitando attività che rendano difficile tornare al compito."),
    ("gruppo", "Come organizzo un lavoro di gruppo?", "Definite un risultato comune, assegnate responsabilità chiare e concordate momenti brevi di verifica."),
    ("domanda", "Come formulo una domanda più precisa?", "Aggiungi contesto, indica il risultato desiderato e specifica eventuali limiti o tentativi già fatti."),
    ("controllo", "Come controllo un lavoro prima di consegnarlo?", "Confrontalo con la richiesta, verifica dati e allegati e rileggilo una volta concentrandoti sugli errori."),
    ("archivio", "Come archivio documenti importanti?", "Usa categorie chiare, nomi coerenti e una copia di sicurezza per i documenti che non puoi perdere."),
    ("routine-serale", "Come preparo il necessario per il giorno dopo?", "Controlla agenda e meteo, raccogli gli oggetti indispensabili e prepara ciò che può essere fatto in anticipo."),
    ("pulizia-rapida", "Come organizzo una pulizia rapida della casa?", "Scegli poche zone visibili, elimina prima il disordine e poi pulisci le superfici dall'alto verso il basso."),
    ("acquisto", "Come confronto due prodotti prima di acquistarli?", "Confronta funzioni realmente utili, costo totale, durata prevista e condizioni di assistenza."),
    ("nota", "Come trasformo un pensiero in una nota utile?", "Scrivi il contesto, l'azione eventuale e la data entro cui vuoi rivedere la nota."),
]

REWRITES = [
    ("rispondi", "Rendi più gentile: Rispondimi subito.", "Potresti rispondermi appena possibile, per favore?"),
    ("errore", "Rendi più professionale: Hai sbagliato tutto.", "Ho rilevato alcuni problemi nel lavoro; possiamo rivederli insieme?"),
    ("ritardo", "Rendi più gentile: Sei in ritardo.", "Ho notato che siamo oltre l'orario previsto: riesci a dirmi quando arriverai?"),
    ("riunione", "Rendi più professionale: Facciamo una riunione domani.", "Possiamo programmare una riunione per domani?"),
    ("allegato", "Rendi più gentile: Hai dimenticato l'allegato.", "Potresti inviare anche l'allegato menzionato, per favore?"),
    ("chiamata", "Rendi più cortese: Chiamami appena puoi.", "Quando hai un momento, potresti chiamarmi?"),
    ("modifica", "Rendi più professionale: Devi modificare questa parte.", "Ti chiederei di rivedere questa parte prima della consegna."),
    ("consegna", "Rendi più gentile: Il lavoro deve essere pronto venerdì.", "Sarebbe possibile completare il lavoro entro venerdì?"),
    ("spiegazione", "Rendi più cortese: Non ho capito niente.", "Potresti spiegarmelo di nuovo con un esempio più semplice?"),
    ("rumore", "Rendi più gentile: Fai meno rumore.", "Potresti abbassare un po' il volume, per favore?"),
    ("attesa", "Rendi più professionale: Sto ancora aspettando.", "Resto in attesa di un aggiornamento quando possibile."),
    ("informazioni", "Rendi più cortese: Voglio sapere quando arriva.", "Potrei sapere quando è prevista la consegna?"),
    ("posto", "Rendi più gentile: Spostati.", "Potresti lasciarmi un po' di spazio, per favore?"),
    ("prestito", "Rendi più cortese: Dammi il libro.", "Potresti prestarmi il libro, per favore?"),
    ("aiuto", "Rendi più professionale: Mi serve aiuto.", "Avrei bisogno di supporto su questa attività."),
    ("correzione", "Rendi più gentile: Correggi questo testo.", "Potresti aiutarmi a correggere questo testo?"),
    ("orario", "Rendi più cortese: Dimmi l'orario.", "Potresti confermarmi l'orario, per favore?"),
    ("conferma", "Rendi più professionale: Dimmi se va bene.", "Potresti confermare se questa soluzione è adeguata?"),
    ("urgenza", "Rendi più gentile: È urgente, fallo ora.", "La richiesta è urgente: riusciresti a occupartene appena possibile?"),
    ("interruzione", "Rendi più cortese: Non interrompermi.", "Potresti lasciarmi terminare il ragionamento, per favore?"),
    ("messaggio-lungo", "Rendi più professionale: Il tuo messaggio è troppo lungo.", "Potresti sintetizzare i punti principali del messaggio?"),
    ("dati", "Rendi più gentile: I dati sono sbagliati.", "Alcuni dati non sembrano corretti: possiamo verificarli?"),
    ("decisione", "Rendi più cortese: Decidi in fretta.", "Riusciresti a comunicarmi la decisione entro breve?"),
    ("presenza", "Rendi più professionale: Devi esserci.", "La tua presenza sarebbe importante per completare la discussione."),
    ("ripetizione", "Rendi più gentile: Me l'hai già detto.", "Grazie, avevo già ricevuto questa informazione."),
    ("rifiuto", "Rendi più cortese: Non mi interessa.", "Ti ringrazio, ma al momento preferisco non proseguire."),
    ("problema", "Rendi più professionale: Questa cosa non funziona.", "La soluzione presenta ancora un problema che deve essere verificato."),
    ("mancanza", "Rendi più gentile: Non hai fatto la tua parte.", "Mi sembra che una parte dell'attività sia ancora da completare: possiamo controllare?"),
    ("priorita", "Rendi più professionale: Fai prima questo.", "Ti chiederei di dare priorità a questa attività."),
]

CLARIFICATIONS = [
    ("computer", "Quale computer dovrei comprare?", "Per consigliarti servono almeno budget, utilizzo principale e preferenze su portabilità e sistema operativo."),
    ("telefono", "Quale telefono mi consigli?", "Qual è il tuo budget e quali aspetti contano di più, per esempio fotocamera, autonomia o dimensioni?"),
    ("viaggio", "Dove dovrei andare in viaggio?", "Da dove parti, in quale periodo, con quale budget e che tipo di esperienza cerchi?"),
    ("libro", "Quale libro dovrei leggere?", "Quali generi ti piacciono e preferisci narrativa, saggistica o qualcosa di breve?"),
    ("film", "Quale film mi consigli?", "Quali generi apprezzi e vuoi un film leggero, intenso, recente o classico?"),
    ("regalo", "Che regalo dovrei comprare?", "Per chi è il regalo, qual è l'occasione e quale budget vuoi rispettare?"),
    ("progetto", "Quale delle mie due idee è migliore?", "Descrivi le due idee e indica obiettivo, vincoli e criteri con cui vuoi confrontarle."),
    ("corso", "Quale corso dovrei seguire?", "Quale competenza vuoi acquisire, qual è il tuo livello e quanto tempo puoi dedicare allo studio?"),
    ("lingua", "Quale lingua straniera dovrei imparare?", "Per quale obiettivo vuoi usarla: lavoro, viaggio, studio o interesse personale?"),
    ("software", "Quale programma devo usare?", "Che attività devi svolgere, su quale sistema operativo e con quali requisiti?"),
    ("formato", "In quale formato devo salvarlo?", "Che tipo di contenuto è e dove dovrà essere aperto, modificato o pubblicato?"),
    ("messaggio", "Mi aiuti a scrivere un messaggio?", "Certo: a chi è rivolto, che cosa vuoi comunicare e quale tono desideri?"),
    ("email", "Mi scrivi un'email?", "Qual è il destinatario, lo scopo dell'email e il tono che vuoi usare?"),
    ("riassunto", "Puoi farmi un riassunto?", "Certo: incolla il testo e dimmi quanto deve essere breve il riassunto."),
    ("traduzione", "Puoi tradurre questo testo?", "Certo: incolla il testo e indica la lingua di destinazione."),
    ("errore", "Perché non funziona?", "Che cosa stai usando, quale risultato ti aspettavi e quale errore o comportamento osservi?"),
    ("codice", "Puoi correggere il mio codice?", "Incolla il codice, l'errore ricevuto e descrivi il comportamento atteso."),
    ("piano", "Mi prepari un piano?", "Qual è l'obiettivo, quanto tempo hai e quali vincoli dobbiamo rispettare?"),
    ("studio", "Quanto devo studiare?", "Per quale materia o esame, entro quale data e qual è il tuo livello attuale?"),
    ("ricetta", "Che cosa posso cucinare?", "Quali ingredienti hai, quanto tempo disponi e ci sono preferenze o alimenti da evitare?"),
    ("organizzazione", "Mi aiuti a organizzarmi?", "Quali attività devi gestire, quali scadenze hai e che cosa ti crea maggiore difficoltà?"),
    ("scelta", "Quale opzione devo scegliere?", "Quali sono le opzioni e quali criteri sono più importanti per te?"),
    ("testo", "Puoi migliorare questo testo?", "Certo: incolla il testo e indica pubblico, obiettivo e tono desiderato."),
    ("presentazione", "Mi aiuti con una presentazione?", "Qual è l'argomento, chi è il pubblico e quanto tempo deve durare?"),
    ("problema", "Mi aiuti a risolvere un problema?", "Descrivi il problema, il risultato desiderato e i tentativi già fatti."),
]

FAKE_COUNTRIES = [
    "Lumeria", "Valdoria", "Norvella", "Estavia", "Belvaria", "Cerenzia", "Dorvania",
    "Elmondia", "Falmeria", "Galdovia", "Istravia", "Jorenia", "Karselia", "Lorvania",
    "Meridia Nova", "Noveria", "Ordelia", "Pavonia Blu", "Querenzia", "Rostavia",
    "Selvaria", "Terenovia", "Uldaria", "Vesperia", "Zafiria", "Brumavia",
    "Caldoria", "Domeria", "Elandria", "Fioravia",
]

FAKE_ENTITIES = [
    ("pianeta", "Qual è il pianeta Zorvax?", "Non riconosco Zorvax come un pianeta noto. Potrebbe essere un nome inventato o esserci un errore?"),
    ("elemento", "Qual è il simbolo chimico del luminio?", "Non riconosco il luminio come elemento chimico. Potresti aver confuso il nome con alluminio?"),
    ("autore", "Chi ha scritto il romanzo Il castello di Norven?", "Non riconosco con certezza questo titolo. Puoi indicare autore, editore o altro contesto?"),
    ("citta", "In quale paese si trova la città di Velmora?", "Non riconosco Velmora come una città reale nota. Puoi verificare il nome o fornire altro contesto?"),
    ("animale", "Che animale è il torvillo azzurro?", "Non riconosco il torvillo azzurro come una specie reale. Potrebbe essere un nome inventato o locale?"),
    ("lingua", "Dove si parla la lingua norvica moderna?", "Non riconosco la norvica moderna come lingua documentata. Puoi controllare il nome?"),
    ("unità", "Quanti metri contiene un valtor?", "Non riconosco il valtor come unità di misura standard. Serve sapere a quale sistema appartiene."),
    ("software", "A che cosa serve il programma Nexalume?", "Non riconosco Nexalume come software noto. Puoi fornire un sito, uno sviluppatore o altro contesto?"),
    ("opera", "Chi ha dipinto Il sole di Braven?", "Non riconosco con certezza quest'opera. Puoi verificare il titolo o indicare il museo?"),
    ("evento", "In quale anno avvenne la battaglia di Lornavia?", "Non riconosco una battaglia storica con questo nome. Potrebbe esserci un errore nella denominazione?"),
    ("moneta", "Qual è la moneta ufficiale della Selvania?", "Non riconosco la Selvania come Stato reale. Puoi verificare il nome?"),
    ("fiume", "Quanto è lungo il fiume Arvelon?", "Non riconosco con certezza un fiume chiamato Arvelon. Puoi indicare il paese o controllare il nome?"),
    ("montagna", "Quanto è alta la montagna Corvessa?", "Non riconosco Corvessa come montagna nota. Puoi fornire la regione o verificare il nome?"),
    ("azienda", "Chi ha fondato l'azienda Veltrixia?", "Non riconosco con certezza questa azienda. Puoi indicare il settore o il sito ufficiale?"),
    ("protocollo", "Che cos'è il protocollo ZTPX-9?", "Non riconosco ZTPX-9 come protocollo standard noto. Puoi indicare il contesto tecnico?"),
    ("formato", "Come apro un file con estensione .nrvx?", "Non riconosco l'estensione .nrvx. Serve sapere quale programma ha creato il file."),
    ("strumento", "Che strumento musicale è il caldofono?", "Non riconosco il caldofono come strumento documentato. Potrebbe essere un nome inventato o regionale?"),
    ("minerale", "Che proprietà ha il minerale elvarite?", "Non riconosco l'elvarite come minerale standard. Puoi verificare il nome o indicare la fonte?"),
    ("teorema", "Che cosa afferma il teorema di Varnelli?", "Non riconosco un teorema standard con questo nome. Puoi indicare l'ambito o una fonte?"),
    ("personaggio", "Chi era l'esploratore Marco Selvar?", "Non riconosco con certezza questa persona storica. Puoi fornire periodo o provenienza?"),
]

TIME_SENSITIVE = [
    ("presidente-italia", "Chi è oggi il Presidente della Repubblica Italiana?", "Le cariche istituzionali cambiano nel tempo: per una risposta aggiornata va consultato il sito ufficiale del Quirinale."),
    ("governo", "Chi guida oggi il governo italiano?", "La guida del governo può cambiare: verifica la composizione aggiornata sul sito ufficiale del Governo italiano."),
    ("commissione-ue", "Chi presiede attualmente la Commissione europea?", "La carica può cambiare nel tempo: verifica il sito ufficiale della Commissione europea."),
    ("onu", "Chi è attualmente il Segretario generale delle Nazioni Unite?", "La carica può cambiare: consulta il sito ufficiale delle Nazioni Unite per il dato aggiornato."),
    ("meteo", "Che tempo farà domani a Roma?", "Il meteo cambia rapidamente: serve una previsione aggiornata per Roma e per la data esatta."),
    ("cambio", "Qual è oggi il cambio tra euro e dollaro?", "Il tasso di cambio varia continuamente: consulta una fonte finanziaria aggiornata."),
    ("borsa", "Quanto vale oggi un'azione di una certa società?", "Il prezzo cambia durante le contrattazioni: servono simbolo della società, mercato e dati aggiornati."),
    ("partita", "Chi ha vinto l'ultima partita della nazionale italiana?", "Serve conoscere data e competizione e consultare risultati sportivi aggiornati."),
    ("popolazione", "Quanti abitanti ha oggi una città?", "La popolazione varia e dipende dalla data e dalla fonte statistica: indica la città e l'anno di riferimento."),
    ("orario-treno", "A che ora parte oggi il prossimo treno?", "Servono stazione di partenza, destinazione, data e orari aggiornati dell'operatore."),
    ("prezzo", "Quanto costa oggi un litro di benzina?", "Il prezzo varia per luogo, data e distributore: serve una fonte locale aggiornata."),
    ("software-versione", "Qual è l'ultima versione disponibile di un programma?", "Le versioni cambiano: indica il programma e verifica la documentazione ufficiale aggiornata."),
]


def id_for_split(group_key: str, variant: int, target: str) -> str:
    for nonce in range(100_000):
        candidate = f"{SOURCE}:{group_key}:v{variant:02d}:{nonce}"
        bucket = v1.fnv1a_64(candidate) % 10_000
        actual = "train" if bucket < 9000 else "validation" if bucket < 9500 else "test"
        if actual == target:
            return candidate
    raise RuntimeError(f"could not assign {group_key} variant {variant} to {target}")


def make_record(group_key: str, category: str, variant: int, question: str, answer: str, split: str) -> dict:
    return {
        "id": id_for_split(group_key, variant, split),
        "source": SOURCE,
        "license": "USER_APPROVED",
        "approval_status": "approved",
        "category": category,
        "messages": [{"role": "user", "content": question}, {"role": "assistant", "content": answer}],
        "provenance": {
            "kind": "controlled_synthetic_balanced_qa",
            "generator": SCRIPT,
            "fact_group": group_key,
            "intended_split": split,
            "variant": variant,
        },
    }


def add_group(rows: list[dict], evaluations: list[dict], group_key: str, category: str,
              questions: list[str], answers: list[str], evaluation_prompt: str) -> None:
    split = v1.group_split(group_key)
    for index, question in enumerate(questions, start=1):
        rows.append(make_record(group_key, category, index, question, answers[(index - 1) % len(answers)], split))
    if split == "train":
        evaluations.append({
            "id": f"manual-eval:{group_key}", "category": category, "fact_group": group_key,
            "prompt": evaluation_prompt, "expected_answer": answers[0],
            "trained_prompt_variants": len(questions),
        })


def add_fact(rows: list[dict], evaluations: list[dict], prefix: str, category: str,
             key: str, question: str, answer: str) -> None:
    add_group(rows, evaluations, f"{prefix}-{key}", category,
              [question, f"Puoi rispondere a questa domanda? {question}",
               f"Rispondi in modo semplice: {question}", f"Dammi una risposta breve: {question}"],
              [answer], f"Mi aiuti con questa domanda? {question}")


def import_v1(rows: list[dict], evaluations: list[dict]) -> None:
    old_rows, old_evaluations = v1.build()
    for old in old_rows:
        provenance = old["provenance"]
        rows.append(make_record(
            provenance["fact_group"], old["category"], provenance["variant"],
            old["messages"][0]["content"], old["messages"][1]["content"],
            provenance["intended_split"],
        ))
    evaluations.extend(old_evaluations)


def add_definitions(rows: list[dict], evaluations: list[dict], prefix: str,
                    category: str, definitions: list[tuple[str, str, str]]) -> None:
    for key, display, answer in definitions:
        add_group(rows, evaluations, f"{prefix}-{key}", category,
                  [f"Che cosa significa il termine «{display}»?",
                   f"Spiega in breve il termine «{display}».",
                   f"Definisci in modo semplice: {display}.",
                   f"Come descriveresti il concetto «{display}»?"],
                  [answer], f"Come spiegheresti il termine «{display}» a un principiante?")


def add_practical(rows: list[dict], evaluations: list[dict]) -> None:
    for key, question, answer in PRACTICAL:
        add_group(rows, evaluations, f"pratico-v2-{key}", "istruzioni_pratiche",
                  [question, f"Dammi un consiglio concreto: {question}",
                   f"Rispondi con pochi passaggi: {question}", f"Mi serve una soluzione semplice. {question}"],
                  [answer], f"Puoi aiutarmi in modo pratico? {question}")


def add_rewrites(rows: list[dict], evaluations: list[dict]) -> None:
    for key, question, answer in REWRITES:
        _, original = question.split(": ", 1)
        add_group(rows, evaluations, f"riscrittura-{key}", "scrittura_e_riscrittura",
                  [question, f"Puoi riscrivere questa frase in modo più cortese? {original}",
                   f"Migliora il tono della frase seguente: {original}",
                   f"Proponi una versione educata della frase: {original}"],
                  [answer], f"Riscrivi con un tono educato: {original}")


def add_clarifications(rows: list[dict], evaluations: list[dict]) -> None:
    for key, question, answer in CLARIFICATIONS:
        add_group(rows, evaluations, f"chiarimento-{key}", "richieste_ambigue",
                  [question, f"Ho bisogno di un consiglio: {question}",
                   f"Puoi aiutarmi? {question}", f"Non so come decidere. {question}"],
                  [answer], f"Prima di rispondere, quali dettagli ti servono? {question}")


def add_unknowns(rows: list[dict], evaluations: list[dict]) -> None:
    for name in FAKE_COUNTRIES:
        group = f"paese-inesistente-{v1.normalise(name).replace(' ', '-')}"
        answers = [
            f"Non riconosco {name} come uno Stato reale. Potrebbe esserci un errore nel nome?",
            f"{name} non risulta un paese reale nelle mie conoscenze. Puoi verificare come si scrive?",
        ]
        add_group(rows, evaluations, group, "incertezza_e_verifica",
                  [f"Qual è la capitale di {name}?", f"Come si chiama la capitale di {name}?",
                   f"Dimmi la capitale di {name}.", f"In quale continente si trova {name}?"],
                  answers, f"Sai indicarmi la capitale di {name}?")
    for key, question, answer in FAKE_ENTITIES:
        add_fact(rows, evaluations, "entita-incerta", "incertezza_e_verifica", key, question, answer)
    for key, question, answer in TIME_SENSITIVE:
        add_fact(rows, evaluations, "dato-temporale", "informazioni_variabili", key, question, answer)


def add_more_math(rows: list[dict], evaluations: list[dict]) -> None:
    divisions = [(12, 3), (18, 2), (20, 4), (24, 6), (30, 5), (32, 4), (36, 6),
                 (40, 8), (42, 7), (45, 5), (48, 6), (54, 9), (56, 7), (63, 9),
                 (64, 8), (72, 8), (81, 9), (90, 10), (96, 12), (100, 4),
                 (120, 6), (144, 12), (150, 5), (200, 8), (225, 9)]
    conversions = [
        ("metri-km-1", "Quanti metri ci sono in un chilometro?", "In un chilometro ci sono 1.000 metri."),
        ("cm-metro-1", "Quanti centimetri ci sono in un metro?", "In un metro ci sono 100 centimetri."),
        ("mm-cm-1", "Quanti millimetri ci sono in un centimetro?", "In un centimetro ci sono 10 millimetri."),
        ("grammi-kg-1", "Quanti grammi ci sono in un chilogrammo?", "In un chilogrammo ci sono 1.000 grammi."),
        ("ml-litro-1", "Quanti millilitri ci sono in un litro?", "In un litro ci sono 1.000 millilitri."),
        ("minuti-ora-1", "Quanti minuti ci sono in un'ora?", "In un'ora ci sono 60 minuti."),
        ("secondi-minuto-1", "Quanti secondi ci sono in un minuto?", "In un minuto ci sono 60 secondi."),
        ("ore-giorno-1", "Quante ore ci sono in un giorno?", "In un giorno ci sono 24 ore."),
        ("giorni-settimana-1", "Quanti giorni ci sono in una settimana?", "In una settimana ci sono 7 giorni."),
        ("mesi-anno-1", "Quanti mesi ci sono in un anno?", "In un anno ci sono 12 mesi."),
    ]
    comparisons = [(14, 9), (25, 52), (101, 99), (37, 37), (64, 46), (120, 210),
                   (305, 350), (88, 78), (1_000, 999), (450, 405), (73, 27), (16, 61),
                   (500, 500), (240, 204), (19, 91)]
    for dividend, divisor in divisions:
        result = dividend // divisor
        key = f"divisione-{dividend}-{divisor}"
        add_group(rows, evaluations, key, "matematica",
                  [f"Quanto fa {dividend} diviso {divisor}?", f"Calcola {dividend} ÷ {divisor}.",
                   f"Qual è il quoziente tra {dividend} e {divisor}?", f"Risolvi la divisione {dividend} : {divisor}."],
                  [f"{dividend} diviso {divisor} fa {result}.", f"Il risultato è {result}."],
                  f"A quanto corrisponde {dividend} diviso {divisor}?")
    for key, question, answer in conversions:
        add_fact(rows, evaluations, "conversione", "matematica", key, question, answer)
    for left, right in comparisons:
        relation = "uguale a" if left == right else "maggiore di" if left > right else "minore di"
        key = f"confronto-{left}-{right}"
        answer = f"{left} è {relation} {right}."
        add_group(rows, evaluations, key, "matematica",
                  [f"{left} è maggiore, minore o uguale a {right}?", f"Confronta i numeri {left} e {right}.",
                   f"Quale relazione c'è tra {left} e {right}?", f"Indica se {left} è sopra, sotto o uguale a {right}."],
                  [answer], f"Come si confrontano {left} e {right}?")


def augment_with_controlled_paraphrases(rows: list[dict]) -> None:
    """Add varied wrappers without changing answers or semantic split.

    These examples teach the model that ordinary conversational lead-ins do
    not change the requested task. They increase linguistic coverage while the
    report continues to expose the smaller number of independent fact groups.
    """
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["provenance"]["fact_group"], []).append(row)

    wrappers = [
        "Vorrei una risposta chiara alla domanda seguente: {question}",
        "Puoi aiutarmi a capire questo punto? {question}",
        "Rispondi con parole semplici: {question}",
        "Ho un dubbio: {question}",
        "Dammi una spiegazione breve e precisa: {question}",
        "Mi serve una risposta diretta: {question}",
    ]
    additions: list[dict] = []
    for group, group_rows in grouped.items():
        canonical = min(group_rows, key=lambda row: row["provenance"]["variant"])
        question = canonical["messages"][0]["content"]
        answer = canonical["messages"][1]["content"]
        category = canonical["category"]
        split = canonical["provenance"]["intended_split"]
        next_variant = max(row["provenance"]["variant"] for row in group_rows) + 1
        for offset, wrapper in enumerate(wrappers):
            additions.append(make_record(
                group, category, next_variant + offset,
                wrapper.format(question=question), answer, split,
            ))
    rows.extend(additions)


def build() -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    evaluations: list[dict] = []
    import_v1(rows, evaluations)
    for key, question, answer in SCIENCE:
        add_fact(rows, evaluations, "scienza-v2", "scienza", key, question, answer)
    for key, question, answer in HUMANITIES:
        add_fact(rows, evaluations, "umanistica-v2", "cultura_umanistica", key, question, answer)
    add_definitions(rows, evaluations, "informatica-v2", "informatica", INFORMATICS)
    add_definitions(rows, evaluations, "quotidiano-v2", "conoscenze_quotidiane", EVERYDAY)
    add_practical(rows, evaluations)
    add_rewrites(rows, evaluations)
    add_clarifications(rows, evaluations)
    add_unknowns(rows, evaluations)
    add_more_math(rows, evaluations)
    augment_with_controlled_paraphrases(rows)
    return rows, evaluations


def validate(rows: list[dict], evaluations: list[dict]) -> dict:
    ids: set[str] = set()
    prompts: set[str] = set()
    split_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    group_splits: dict[str, str] = {}
    max_user = 0
    max_assistant = 0
    for row in rows:
        if row["id"] in ids:
            raise RuntimeError(f"duplicate ID: {row['id']}")
        ids.add(row["id"])
        question = row["messages"][0]["content"]
        answer = row["messages"][1]["content"]
        prompt_key = v1.normalise(question)
        if prompt_key in prompts:
            raise RuntimeError(f"duplicate prompt: {question}")
        prompts.add(prompt_key)
        if "…" in question or "…" in answer:
            raise RuntimeError(f"ellipsis is not allowed: {row['id']}")
        if len(question) > 260 or len(answer) > 360:
            raise RuntimeError(f"record too long: {row['id']}")
        provenance = row["provenance"]
        group = provenance["fact_group"]
        split = provenance["intended_split"]
        if group in group_splits and group_splits[group] != split:
            raise RuntimeError(f"split leakage for group: {group}")
        group_splits[group] = split
        split_counts[split] += 1
        category_counts[row["category"]] += 1
        max_user = max(max_user, len(question))
        max_assistant = max(max_assistant, len(answer))
    eval_prompts = [v1.normalise(item["prompt"]) for item in evaluations]
    if len(eval_prompts) != len(set(eval_prompts)):
        raise RuntimeError("duplicate manual evaluation prompt")
    if prompts.intersection(eval_prompts):
        raise RuntimeError("manual evaluation prompt leaked into source")
    return {
        "records": len(rows), "semantic_groups": len(group_splits),
        "manual_evaluations": len(evaluations),
        "split_counts": dict(sorted(split_counts.items())),
        "category_counts": dict(sorted(category_counts.items())),
        "max_user_characters": max_user, "max_assistant_characters": max_assistant,
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8") as destination:
        for row in rows:
            destination.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    source_path = args.output_directory / "italiano-simple-qa-v2.source.jsonl"
    evaluation_path = args.output_directory / "italiano-simple-qa-v2.manual-eval.jsonl"
    report_path = args.output_directory / "italiano-simple-qa-v2.report.json"
    for path in (source_path, evaluation_path, report_path):
        if path.exists():
            raise SystemExit(f"refusing to overwrite existing output: {path}")
    rows, evaluations = build()
    report = {
        "schema": "italiano-simple-qa-v2-report",
        "source": SOURCE,
        "policy": {
            "current_facts": "answer with verification guidance, do not memorize office holders",
            "systems": "none",
            "truncation": "forbidden",
            "split_unit": "semantic_group",
            "manual_eval_in_training": False,
        },
        **validate(rows, evaluations),
    }
    args.output_directory.mkdir(parents=True, exist_ok=True)
    write_jsonl(source_path, rows)
    write_jsonl(evaluation_path, evaluations)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
