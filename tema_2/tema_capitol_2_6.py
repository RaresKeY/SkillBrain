# Tema 2 - Capitolul 2.6: Exercițiu bonus – Conversia timpului
#     Cere de la utilizator un număr de secunde și afișează:
#     Câte ore, minute și secunde reprezintă
#     Exemplu:
#         Input: 3665
#         Output: 1 oră, 1 minut, 5 secunde

print(str(s := int(input("Secunde: "))).removeprefix(str(s)),
      "Output:",
      h := s // 3600, "oră" if h == 1 else "ore", ":",
      m := (s // 60) % 60, "minut" if m == 1 else "minute", ":",
      sec := s % 60, "secundă" if sec == 1 else "secunde")  

