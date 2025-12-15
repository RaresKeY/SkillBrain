from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QButtonGroup, QFrame, QSpacerItem, QSizePolicy
)
from PySide6.QtCore import Qt
from quiz_maker import Quiz
import sys


class QuizApp(QWidget):
    def __init__(self, quiz_name="History Quiz"):
        super().__init__()
        self.setWindowTitle("📚 Quiz App")
        self.setMinimumSize(500, 400)
        self.setStyleSheet("""
            QWidget {
                background-color: #1e1e2f;
                color: #f2f2f2;
                font-family: 'Segoe UI', sans-serif;
            }
            QLabel {
                font-size: 18px;
                padding: 6px;
            }
            QPushButton {
                background-color: #3a3a4f;
                border: 2px solid #5a5a7f;
                border-radius: 8px;
                padding: 8px;
                font-size: 16px;
                color: #ffffff;
            }
            QPushButton:hover:!disabled {
                background-color: #5a5a7f;
            }
            QPushButton:disabled {
                opacity: 0.6;
            }
        """)

        self.layout = QVBoxLayout(self)
        self.layout.setAlignment(Qt.AlignTop)
        self.layout.setSpacing(10)

        self.quiz = Quiz(quiz_name)
        self.quiz.load_quiz()
        self.current_index = 0
        self.score = 0

        # Question label
        self.question_label = QLabel("", alignment=Qt.AlignCenter)
        self.question_label.setWordWrap(True)
        self.layout.addWidget(self.question_label)

        # Divider line
        self.divider = QFrame()
        self.divider.setFrameShape(QFrame.HLine)
        self.divider.setStyleSheet("color: #555; margin: 10px 0;")
        self.layout.addWidget(self.divider)

        # Option buttons
        self.button_group = QButtonGroup(self)
        self.buttons = []
        for _ in range(4):
            btn = QPushButton()
            btn.clicked.connect(self.check_answer)
            self.layout.addWidget(btn)
            self.buttons.append(btn)
            self.button_group.addButton(btn)

        # Correct answer label (hidden until answered)
        self.correct_label = QLabel("")
        self.correct_label.setAlignment(Qt.AlignCenter)
        self.correct_label.setStyleSheet("font-size: 16px; color: #8aff8a; margin-top: 8px;")
        self.layout.addWidget(self.correct_label)

        # Spacer
        self.layout.addItem(QSpacerItem(0, 20, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # Next button
        self.next_button = QPushButton("Next ➡️")
        self.next_button.clicked.connect(self.next_question)
        self.next_button.setEnabled(False)
        self.layout.addWidget(self.next_button, alignment=Qt.AlignCenter)

        self.load_question()

    def load_question(self):
        q = self.quiz.sections[self.current_index]
        self.question_label.setText(f"Q{self.current_index + 1}: {q.question}")
        self.correct_label.clear()
        self.next_button.setEnabled(False)

        for i, btn in enumerate(self.buttons):
            if i < len(q.options):
                btn.setText(q.options[i])
                btn.setEnabled(True)
                btn.show()
                btn.setStyleSheet("background-color: #3a3a4f; border: 2px solid #5a5a7f; border-radius: 8px;")
            else:
                btn.hide()

    def check_answer(self):
        sender = self.sender()
        q = self.quiz.sections[self.current_index]
        idx = self.buttons.index(sender)

        # Disable all buttons
        for b in self.buttons:
            b.setEnabled(False)

        # Mark correct and incorrect visually
        for i, b in enumerate(self.buttons):
            if i in q.correct:
                b.setStyleSheet("background-color: #3c8c3c; color: white; border-radius: 8px;")
            elif b is sender:
                b.setStyleSheet("background-color: #8c3c3c; color: white; border-radius: 8px;")

        if idx in q.correct:
            self.score += 1

        correct_opts = ", ".join(q.options[i] for i in q.correct)
        self.correct_label.setText(f"✅ Correct answer: {correct_opts}")
        self.next_button.setEnabled(True)

    def next_question(self):
        self.current_index += 1
        if self.current_index < len(self.quiz.sections):
            self.load_question()
        else:
            self.question_label.setText(f"🎉 Quiz Finished! Final Score: {self.score}/{len(self.quiz.sections)}")
            for b in self.buttons:
                b.hide()
            self.next_button.setText("Close")
            self.next_button.clicked.disconnect()
            self.next_button.clicked.connect(self.close)
            self.correct_label.clear()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = QuizApp("History Quiz")
    window.show()
    sys.exit(app.exec())
