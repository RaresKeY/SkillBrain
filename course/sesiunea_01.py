# 01
print("Hello World!!")
print()

# 02
oras, varsta, nume = "Bucuresti", 20, "Ion"
print("Numele meu este", nume, "si am", varsta, "ani. Locuiesc in", oras)
print()

# 03
text, number_int, number_float, boolean = "Text", 42, 3.14, True
print(type(text), type(number_int), type(number_float), type(boolean), sep="\n")
print()

# Create hello world
print()

# 04
a, b = 10, 5
print("Adunare:", a + b)
print("Scădere:", a - b)
print("Înmulțire:", a * b)
print("Înpărțire:", a / b)
print("Înpărțire întreagă:", a // b)
print("Restul împărțirii:", a % b)
print("Putere:", a**b)
print()

# 05
print(
    "Salut ",
    input("Cum te cheama? "),
    "!\n",
    "La anul vei avea ",
    int(input("Cati ani ai? ")) + 1,
    " de ani",
    "\nOrasul tau este: ",
    input("Care e orasul tau? "),
    sep="",
)
print()

# 06
nume, varsta, oras = "Ana", 25, "Cluj"
print(f"Salut {nume}! Ai {varsta} ani si locuiesti in {oras}.")
print()

# 07
print(int(float(input("Numar 1: "))) + int(float(input("Numar 1: "))))
print()

# 08
nume = "Ana"
print("a" in nume)
print()

# 09
print(
    len("Budget"),
    max(12, 2, 11),
    min(8, 4, 5),
    abs(-232.23),
    round(5.625434, 2),
    type(float("nan")),
)

# 10
name, age, city = input("Nume: "), input("Varsta: "), input("Localitate:")
print(
    f"Salut {name} din {city}. Peste 10 ani vei avea {int(age) + 10}.",
    "Ai" if "a" in name else "Nu ai",
    "litera 'a' in nume.",
)
print()

# 11
name, oras, age, job = "George", "Bucuresti", "22", "Programmer"
print("-" * 30)
print("Nume:".ljust(9), name)
print("Oras:".ljust(9), oras)
print("Varsta:".ljust(9), f"{age} ani")
print("Ocupatie:".ljust(9), job)
print("-" * 30)
