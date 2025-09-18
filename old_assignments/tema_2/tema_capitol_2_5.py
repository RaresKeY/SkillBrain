# Tema 2 - Capitolul 2.5: Calculul vârstei
#     Cere de la utilizator anul nașterii și calculează vârsta actuală.
#     Folosește anul curent.

import datetime as datetime

print("Varsta:", datetime.date.today().year - int(input("Anul Nasteri: ")), "ani")
