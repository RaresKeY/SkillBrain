# Tema 2 - Capitolul 2.3: Comparare numere
#     Cere două numere de la utilizator
#     Afișează dacă primul este mai mare, mai mic sau egal cu al doilea
#     (3 printuri)

print(str(n1 := int(input("Numar 1: "))).removeprefix(str(n1)),
      str(n2 := int(input("Numar 2: "))).removeprefix(str(n2)),
      'mai mare' if n1 > n2 
      else str(print('egal')).removeprefix("None") if n1 == n2 
      else str(print('mai mic')).removeprefix("None"),
      sep='') # 3️⃣ 🖨️ 😎

