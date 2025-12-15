class Vision:
    def __init__(self): 
        print("Vision")
        super().__init__()

class Hearing:
    def __init__(self): 
        print("Hearing")
        super().__init__()

class Event(Vision, Hearing): pass
class Other(Hearing, Vision): pass

Event()
# Vision → Hearing

print()

Other()
# Hearing → Vision


# walk MRO manually (bruteforce) for non-cooperative classes
# one unified value for each sense, event focused/originated
# class Event(Vision, Hearing, Smell, Touch, Taste):
#     def __init__(self):
#         for cls in Event.mro()[1:-1]:  # skip Event and object
#             init = cls.__dict__.get("__init__")
#             if init:
#                 print(f"Calling {cls.__name__}.__init__")
#                 init("dummy")
