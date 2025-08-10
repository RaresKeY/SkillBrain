# Tema 2 - Capitolul 2.1: Calculator simplu
#     Cere două numere de la utilizator
#     Afișează suma, diferența, produsul, împărțirea normală, împărțirea întreagă și restul împărțirii

print(str(first_operand := int(input("Numar 1: "))).removeprefix(str(first_operand)),
      str(second_operand := int(input("Numar 2: "))).replace(str(second_operand), ""),
      '\nsuma:'.ljust(22),                  first_operand + second_operand,
      '\ndiferența:'.ljust(22),             first_operand - second_operand,
      '\nprodusul:'.ljust(22),              first_operand * second_operand,
      '\nîmpărțirea normală:'.ljust(22),    first_operand / second_operand if second_operand else "NaN",
      '\nîmpărțirea întreagă:'.ljust(22),   first_operand //second_operand if second_operand else "NaN",
      '\nrestul împărțirii:'.ljust(22),     first_operand % second_operand if second_operand else float('nan'),
      sep='')

