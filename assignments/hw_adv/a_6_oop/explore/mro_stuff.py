class Vision:
    def __init__(self, vision_event=None, bearing=None, elevation=None, distance=None, **kwargs):
        self.vision_event = vision_event
        self.bearing = bearing
        self.elevation = elevation
        self.distance = distance
        super().__init__(**kwargs)

class Hearing:
    def __init__(self, hearing_event=None, l_intens=None, l_reverb=None, r_intens=None, r_reverb=None, **kwargs):
        self.hearing_event = hearing_event
        self.l_intens = l_intens
        self.l_reverb = l_reverb
        self.r_intens = r_intens
        self.r_reverb = r_reverb
        super().__init__(**kwargs)

class Touch:
    def __init__(self, touch_event=None, intensity=None, location=None, **kwargs):
        self.touch_event = touch_event
        self.intensity = intensity
        self.location = location
        super().__init__(**kwargs)

class Smell:
    def __init__(self, smell_event=None, smell_intensity=None, **kwargs):
        self.smell_event = smell_event
        self.smell_intensity = smell_intensity
        super().__init__(**kwargs)

class Taste:
    def __init__(self, taste_event=None, taste_intensity=None, **kwargs):
        self.taste_event = taste_event
        self.taste_intensity = taste_intensity
        super().__init__(**kwargs)

# unified class via multiple inheritance
class EventOne(Vision, Hearing, Touch, Smell, Taste):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

# Example
e = EventOne(
    vision_event="flash", bearing=15, distance=42,
    hearing_event="boom", l_intens=0.9,
    touch_event="hit", location="arm",
    smell_event="smoke", smell_intensity=5,
    taste_event="bitter", taste_intensity=2
)

print(e.__dict__)