# 📈 Tema

## [ ] I. Adapteaza-ti un exercitiu la alegere din temele trecute, implementand functii.

Foloseste-ti imaginatia, am putea pune fiecare linie intr-o functie separata, dar Arta este sa distingem acele parti cheie compuse din secvente de actiuni care compun o Actiune mai mare - Re-utilizabila, Organizata, Modulara

Ex: Functie pentru meniu, functie pentru adaugare(), functie pentru filtrare, cautare, etc.

## [x] II. Funcție care primește text → returnează:
    nr. de cuvinte
    cel mai lung cuvânt
    dacă apare „python”

## [x] III.Tema Opțională (Nivel Avansat) – Blackjack

Scop: construiește motorul de joc Blackjack folosind funcții.

### Programul principal trebuie sa apeleze doar 2 functii:

player = seat_player(name, budget)
run_blackjack(player)

### Restul organizam din functii:

Programul trebuie să ruleze pe pași, oprindu-se unde este nevoie de input de la utilizator (ex. când decide dacă dă “Hit” sau “Hold”).

### Fa mai intai Planul pe hârtie si/sau in minte:  
#### [x] I. Deal Cards – Programul împarte cărți.
- Jucătorul își vede cărțile.  
- Casa (computerul) își ascunde cărțile, dar le memorează.  
- Algoritm (funcție) de împărțire a cărților – ne folosim de random pentru a genera numărul cărții și simbolul. Pentru cine vrea să meargă mai departe – algoritmul ar trebui, ca în realitate, să țină cont de cărțile care au fost deja date.

Can you Imagine a way to do this? 🤔


#### [x] II. User Action – Jucătorul poate doar:
- Hit (primește o carte nouă)
- Hold (stă pe ce are).
- Bugetul este simplificat: bet fix de 100.


#### [x] III. Bust Check – dacă utilizatorul trece de 21, pierde instant.  

#### [x] IV. Reveal House Cards – odată ce utilizatorul dă Hold sau Bust, se afișează cărțile Casei.

#### [x] V. House Strategy – Casa decide dacă: (implemented different logic)
- Ține (dacă are deja > valoarea jucătorului).
- Sau cere cărți până depășește valoarea jucătorului ori până trece de 21.
- Algoritmul Casei – facem asta cu un if-else bazat pe praguri simple pentru a decide dacă să stea sau să mai tragă o carte (dacă are > valoarea utilizatorului – stă, nu are sens să mai tragă deoarece a câștigat deja; altfel trage până când depășește valoarea utilizatorului (câștigă) sau trece peste 21)

#### [x] VI. Rezultat final – Afișează cine a câștigat și actualizează bugetul jucătorului.

### Extra Optional:
#### [ ] VII. Basic Graphics
- [ ] Card overlay (number + symbol)
- [ ] Card images background (AI Gen flow, check on process established)
- [ ] Table Texture background (AI Gen flow)
- [ ] Dealer Character 2D (AI Gen flow)
    - [ ] Multiple poses (see animation)
- [ ] Hand 2D (AI Gen flow)

#### [ ] VIII. Basic Sound Design 
- [ ] Music
- [ ] Sound Effects
- [ ] Voice lines (AI Gen - MelloTTS + whisper check flow)

#### [ ] IX. Basic Physics
- [ ] Card throw button
- [ ] Card collisions
- [ ] Card moves other cards on top slightly, paper

#### [ ] X. Basic Animations
- [ ] Basic 2D Sprite stretcha and pull
- [ ] Dealer basic switch pic animations style
- [ ] Basic 3d
- [ ] Dealing flow
- [ ] Paper rigging

#### [ ] XI. 


> Vizualizeaza clar Flow-ul Algoritmului la fiecare pas.
>
> Apoi, pune-l in cod.
>
> Pas - cu pas.