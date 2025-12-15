import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

class Section():
    def __init__(self, question: str, options: list[str], correct: list[int]):
        self.question: str = question
        self.options: list[str] = options
        self.correct: list[int] = correct


class Quiz():
    def __init__(self, name, file_path = Path("quizes.json")):
        self.file_path: Path = file_path if file_path.is_absolute() else (BASE_DIR / file_path)
        self.name: str = name
        self.sections: list[Section] = []

    def quick_add(self, question, options: list, correct: list[int]):
        self.sections.append(Section(question, options, correct))

    def load_quiz(self, name = None):
        if name:
            self.name = name

        if not self.file_path.exists():
            raise FileNotFoundError(f"{self.file_path} not found")

        with open(self.file_path, 'r') as f:
            data = json.load(f)

        quiz = data.get(self.name)

        if not quiz:
            raise ValueError(f"No data found for {self.name}")

        self.sections = [
            Section(q["question"], q["options"], q["correct"])
            for q in quiz
        ]

    def save_quiz(self):
        if not self.sections:
            raise ValueError("No sections to save.")

        if self.file_path.exists():
            with open(self.file_path, 'r') as f:
                data = json.load(f)
        else:
            data = dict()

        data[self.name] = [
            {"question": s.question, "options": s.options, "correct": s.correct}
            for s in self.sections
        ]

        with open(self.file_path, 'w') as f:
            json.dump(data, f, indent=2)


quiz = Quiz("History Quiz")
quiz.quick_add("Who discovered America?", ["Columbus", "Newton", "Einstein"], [0])
quiz.quick_add("Year of WW2 start?", ["1914", "1939", "1945"], [1])
quiz.save_quiz()

# Load it back
loaded = Quiz("History Quiz")
loaded.load_quiz()
for s in loaded.sections:
    print(f"{s.question}\n  → Correct: {', '.join(s.options[i] for i in s.correct)}\n")


