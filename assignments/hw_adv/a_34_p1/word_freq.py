import os
from pathlib import Path
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import random

os.chdir(Path(__file__).parent)

def word_freq(text: str, cuttoff: int = 3) -> dict:
    words = text.strip().split()
    words = [word.strip('.,!?";()—').lower() for word in words]

    words = [w for w in words if len(w) > cuttoff]

    freq = {}
    for word in words:
        freq[word] = freq.get(word, 0) + 1
    return freq

def soft_deep_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    # A palette of soft teals, deep purples, and muted blues
    colors = [
        "hsl(190, 45%, 70%)",  # Soft Teal
        "hsl(210, 50%, 60%)",  # Muted Blue
        "hsl(260, 40%, 50%)",  # Deep Lavender
        "hsl(280, 50%, 40%)",  # Soft Deep Purple
        "hsl(170, 40%, 45%)",  # Deep Seafoam
    ]
    return random.choice(colors)


if __name__ == "__main__":
    # test_text = "This is a test. This test is only a test."

    with open("news.txt", "r") as f:
        test_text = f.read()
    
    freq = word_freq(test_text)
    for word in freq:
        print(f"{word}: {freq[word]}")

    wc = WordCloud(
        width=1600, 
        height=800, 
        background_color='#121212', 
        max_words=200,
        prefer_horizontal=0.7,
        font_step=1,
        relative_scaling=0.5,
        color_func=soft_deep_color_func
    ).generate_from_frequencies(freq)

    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")

    filename = "wordcloud_output.png"
    plt.savefig(
        filename, 
        dpi=300, 
        bbox_inches='tight', 
        pad_inches=0, 
        facecolor='#1a1a1a'
    )

    print(f"File saved successfully as: {filename}")

    plt.show()