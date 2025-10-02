class Story:
    def __init__(self):
        self.sequence = [] # 5 senses sequence
    pass

class StoryElement(Story):
    pass

class Brain:
    def __init___(self):
        self.stories = []

        self.perception_cues = {
            "vision": [],  # Vision(),
            "hearing": [], # Sound(),
            "smell": [],   # Smell(),
            "touch": [],   # Touch(),
            "taste": [],   # Taste()
        }

        self.focus = None

    def record_story(self):
        self.stories.append(Story())

    def change_focus(self, focus):
        self.focus = self.perception[focus]

# locator
class Vision:
    def __init__(self, event,
                 bearing=None, # -90 90
                 elevation=None,
                 distance=None,
                 *args, **kwargs):
        self.__event = event
        self.__bearing = bearing # high res in focus
        self.__elevation = elevation # high res in focus
        self.__distance = distance # med res in focus

vision = {
    "event": "leaf",
    "position": "front"
}

class Vision:
    def __init__(self, event, position):
        self.event = event
        self.position = position

my_vision = Vision("leaf", "front")

print(my_vision.__dict__)

# locator
class Hearing:
    def __init__(self, event,
                 l_intens=None, l_reverb=None,
                 r_intens=None, r_reverb=None,
                 *args, **kwargs):
        self.__event = event
        self.__testing = 0
        # computed from left/right
        # self.breaing = bearing
        # self.elevation = elevation
        # self.intensity = intensity

# decider
class Touch:
    def __init__(self, event, intensity=None, location=None, *args, **kwargs):
        self.__event = event
        self.__intensity = intensity
        self.__location = location

# indicator
class Smell:
    def __init__(self, event, intensity=None, *args, **kwargs):
        self.__event = event
        self.__intensity = intensity

# indicator
class Taste:
    def __init__(self, *args, **kwargs):
        self.__event = kwargs["event"]
        self.__intensity = kwargs["intensity"]

# one unified value for each sense, event focused/originated
class Event:
    def __init__(self, vision=None, hearing=None, touch=None, smell=None, taste=None):
        self.vision = vision
        self.hearing = hearing
        self.touch = touch
        self.smell = smell
        self.taste = taste

vision = Vision(event="flash", distance=42, bearing=15)
hearing = Hearing(event="boom", l_intens=0.9)
smell   = Smell(event="smoke", intensity=5)

e = Event(vision=vision, hearing=hearing, smell=smell)

print(e.__dict__)

focus_events = [(Hearing("barking", distance="far")),
                (Smell("flower scent"), Taste("autumn")),
                (Hearing("wind", "front-left"), Hearing("rustling leaves")),
                (Vision("dog", "front", "close"), Hearing("barking")),
                (Vision("tail wag", distance="close"), Touch("fur", "hand"))]

# --- 1. Show MRO / super() chain ---
print("MRO (super() chain):")
for cls in Event.mro():
    print(" ", cls.__name__)

# --- 2. Collect attributes from all classes in MRO ---
attr_sources = {}
for cls in Event.mro():
    for name, val in cls.__dict__.items():
        if name not in attr_sources:
            attr_sources[name] = []
        attr_sources[name].append(cls.__name__)

print("\nAll attributes (with overlaps):")
for name, sources in attr_sources.items():
    print(f"{name} -> defined in {sources}")

# --- 3. What an instance actually has after init ---
e = Event()
print("\nInstance __dict__ after init:")
print(e.__dict__)

# class methods
print(Event.__dict__)

