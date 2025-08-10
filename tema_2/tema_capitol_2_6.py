# Tema 2 - Capitolul 2.6: Exercițiu bonus – Conversia timpului
#     Cere de la utilizator un număr de secunde și afișează:
#     Câte ore, minute și secunde reprezintă
#     Exemplu:
#         Input: 3665
#         Output: 1 oră, 1 minut, 5 secunde

print(str(total_seconds := int(input("Secunde: "))).removeprefix(str(total_seconds)),
      "Output:",
      hours := total_seconds // 3600, "oră" if hours == 1 else "ore", ":",
      minutes := (total_seconds // 60) % 60, "minut" if minutes == 1 else "minute", ":",
      seconds := total_seconds % 60, "secundă" if seconds == 1 else "secunde")  

