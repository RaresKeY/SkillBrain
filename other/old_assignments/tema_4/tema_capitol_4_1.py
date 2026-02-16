# Tema 4
# Scrie un program care:
# Cere utilizatorului să introducă un PIN.
# Transformă parola în litere mici (lower()).
# Evaluează parola astfel:
# Parolă slabă: dacă are mai puțin de 6 caractere.
# Parolă medie: dacă are 6-10 caractere.
# Parolă sigură: dacă are mai mult de 10 caractere.
# Afișează mesajul corespunzător: „Parolă slabă”, „Parolă medie” sau „Parolă sigură”.

pin_input = input("PIN: ").lower()
pin_length = len(pin_input)

if pin_length < 6:
    print("Parolă slabă”")
elif pin_length <= 10:
    print("Parolă medie")
else:
    print("Parolă sigură")
