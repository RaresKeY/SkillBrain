# Tema 2 - Capitolul 2.2: Perimetrul și aria unui dreptunghi
#     Cere lungimea și lățimea de la utilizator
#     Calculează și afișează perimetrul și aria

print(f"{'Perimetrul:':11} {2 * (l := int(input('lungimea: '))) + 2 * (w := int(input('latimea: ')))}",
      f"\n{'Aria:':11} {l * w}")
      