## 📈 Tema 4: Colectii - Avansati


### [x] **I. Smart Playlist Manager**
- Listă cu melodii (dicționare: titlu, artist, gen, durata, rating).
- Operații: afișare, filtrare după gen, sortare după rating/durată, calcul timp total.
- Algoritm „mood-based”: utilizatorul alege dispoziția („party”, „study”, „relax”) → programul recomandă melodii potrivite.

### [x] **II. Smart Shopping & Budget Manager**
- Dicționar produse: preț.  
- Operații: adăugare, ștergere, afișare total, sortare, filtrare produse > 50 lei.  
- Algoritm „buget”: utilizatorul setează un buget → dacă depășește, programul sugerează ce produse să scoată sau alternative mai ieftine.  

### [x] **III. Smart Activity Assistant**
**Nevoie Client: Aplicație inteligenta pentru managementul prietenilor și hobbyurilor – „Rămâi conectat mereu cu oamenii apropiați”.**  

**Scope:**
- Listă cu prieteni (dicționare: nume, telefon, vârstă, ultim_contact, hobbyuri, distanță, conexiune).
- Operații: adăugare/ștergere/modificare prieteni, afișare hobbyuri unice.
- Algoritm „iesi la un hobby”:

I. Utilizatorul cere un hobby → programul caută prietenii potriviți, calculează scor în funcție de: 
- hobby comun (+)
- contact recent (+)
- distanță (–)
- conexiune (+)

II. Afișează cei mai recomandați.  

III. Simulare interacțiuni: ieșiri la hobby → actualizează ultim_contact și conexiune.  

---

#### Ramaneti ancorati in lumea reala - incercati s-o transpuneti in Structuri de Date.

Scopul este sa transpunem Realitatea in Cod.

"Program în care putem gestiona prietenii și activitățile. În plus, putem cere un hobby (ex. avem chef de fotbal), iar algoritmul parcurge dicționarul după o logică similară cu cea din realitate: caută prietenii care au acel hobby, analizează ultimul contact (cu cât e mai recent, cu atât crește punctajul), analizează distanța (scade punctajul dacă e mare), analizează nivelul de conexiune (crește punctajul dacă e ridicat) și ne aduce persoana cu cele mai mari șanse să ni se potrivească.

Putem merge cu unul sau mai mulți prieteni la un hobby (executăm un hobby → actualizăm ultim_contact pentru acel prieten/prieteni). Conexiunea crește cu un mic procent, dar dacă nu ne vedem mult timp, conexiunea scade."

---

#### *Ghidaj General*

*Meniul se gestioneaza ca si la PIN - dintr-o bucla infinita.

*Orice proiect se poate realiza strict folosind lucrurile pe care le-am invatat pana acum. (fara functii, librarii, etc).

*Totusi, odata ce intelegeti nivelul de baza si puteti urmari clar acel "Flow Of the Script" de care am discutat ieri, simtiti-va liberi sa explorati si sa duceti codul la urmatorul nivel: functii, librarii, etc.