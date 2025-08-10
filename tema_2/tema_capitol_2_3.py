# Tema 2 - Capitolul 2.3: Comparare numere
#     Cere două numere de la utilizator
#     Afișează dacă primul este mai mare, mai mic sau egal cu al doilea
#     (3 printuri)

print(str(reference_number := int(input("Numar 1: "))).removeprefix(str(reference_number)),
      str(comparison_number := int(input("Numar 2: "))).removeprefix(str(comparison_number)),
      'mai mare' if reference_number > comparison_number 
      else str(print('egal')).removeprefix("None") if reference_number == comparison_number 
      else str(print('mai mic')).removeprefix("None"),
      sep='') # 3️⃣ 🖨️ 😎

