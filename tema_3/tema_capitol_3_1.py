# Tema 3

# Scrie un program care:

# Are deja salvate două informații: numele de utilizator și parola corectă.
# Cere utilizatorului să introducă numele de utilizator și parola de la tastatură.

# Verifică următoarele situații:
#     Dacă numele de utilizator introdus este corect și parola introdusă este corectă → afișează "Acces permis".
#     Dacă doar una dintre cele două este corectă (numele de utilizator sau parola) → afișează "User/Password incorect".
#     Dacă ambele sunt greșite → afișează "Acces respins".

import hmac
import hashlib
import getpass

# Stored from "registration"
# ---------------------------------------------------------------------------------------
username_stored = "plaintext_username"
salt_bytes = bytes.fromhex("7f3a2b6d9c1e4f8a6b3c0d1e2f4a5b6c")
iters = 100_000
password_stored = hashlib.pbkdf2_hmac('sha256', b"plaintext_password", salt_bytes, iters)
# ---------------------------------------------------------------------------------------
# I wouldn't put password as plaintext in source code in a real implemention IMO

username_input = input("Username: ")
password_ok = hmac.compare_digest(hashlib.pbkdf2_hmac('sha256', getpass.getpass("Password: ").encode('utf-8'), salt_bytes, iters), password_stored)

if username_input == username_stored and password_ok:
    print("Acces permis")
elif username_input != username_stored and not password_ok:
    print("Acces respins")
else:
    print("User/Password incorect")



