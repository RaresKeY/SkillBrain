# [x] Assignment I - Adaptation

# Original instruction:
#   Adapt one exercise of your choice from the previous assignments
#   by implementing functions (e.g., menu, add, filter, search).

# My adaptation:
#   I refactored my previous assignements code into a reusable
#   function library (helper module), extracting common actions
#   into modular, reusable functions.


from painting_on_water import vinput

if __name__ == "__main__":
    x = vinput("Enter a number: ", pattern=r"^\d+$")
    print(x)