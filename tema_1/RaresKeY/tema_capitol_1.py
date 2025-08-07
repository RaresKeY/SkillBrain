# Într-un fișier Python numit tema_capitol_1.py, 
# scrieți un program care să conțină 4 variabile cu valori introduse de utilizator folosind funcția input(). 
# În funcția input() trebuie scrise întrebări adresate utilizatorului, pe care să le ajustați astfel încât 
# răspunsurile să corespundă tipurilor variabilelor: una de tip str (text), una convertită cu int(), una cu float() 
# și una cu bool(). Programul va afișa apoi fiecare variabilă împreună cu tipul acesteia, folosind funcțiile print() și type(). 
# Enunțul temei trebuie scris ca un comentariu la începutul fișierului.


# Succes!


print("Nume". ljust(7), (n := input("Nume: ")).ljust(6),                               ' | ', type(n), '\n',
      "Ani".  ljust(7), str(a := int(input("Ani: "))).ljust(6),                        ' | ', type(a), '\n',
      "Medie".ljust(7), str(m := float(input("Medie: "))).ljust(6),                    ' | ', type(m), '\n',
      "Activ".ljust(7), str(c := input("Activ(Da/Nu): ").casefold() == "da").ljust(6), ' | ', type(c),
      sep='')