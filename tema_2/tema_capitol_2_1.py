# Tema 2 - Capitolul 2.1: Calculator simplu
#     Cere două numere de la utilizator
#     Afișează suma, diferența, produsul, împărțirea normală, împărțirea întreagă și restul împărțirii

print(str(n1 := int(input("Numar 1: "))).removeprefix(str(n1)),
      str(n2 := int(input("Numar 2: "))).replace(str(n2), ""),
      '\nsuma:'.ljust(22),                  n1 + n2,
      '\ndiferența:'.ljust(22),             n1 - n2,
      '\nprodusul:'.ljust(22),              n1 * n2,
      '\nîmpărțirea normală:'.ljust(22),    n1 / n2 if n2 else "NaN",
      '\nîmpărțirea întreagă:'.ljust(22),   n1 //n2 if n2 else "NaN",
      '\nrestul împărțirii:'.ljust(22),     n1 % n2 if n2 else float('nan'),
      sep='')

