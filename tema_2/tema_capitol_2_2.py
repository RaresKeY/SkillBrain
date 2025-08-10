# Tema 2 - Capitolul 2.2: Perimetrul și aria unui dreptunghi
#     Cere lungimea și lățimea de la utilizator
#     Calculează și afișează perimetrul și aria

print(f"{'Perimetrul:':11} {2 * (length := int(input('lungimea: '))) + 2 * (width := int(input('latimea: ')))}",
      f"\n{'Aria:':11} {length * width}")
      