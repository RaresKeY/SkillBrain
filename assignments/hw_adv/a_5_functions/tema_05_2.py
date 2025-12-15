# [x] II. Function that takes a text → returns:
#   the number of words
#   the longest word
#   whether `"python"` appears


def get_str_data(sentence: str) -> dict:
    """Analyzes text and returns statistics about words.

    Args:
        sentence: Input text string to analyze.

    Returns:
        dict: Dictionary containing the following statistics:
            \n- word_count (int): Total number of words in the sentence
            \n- longest_word (str): Word with the most characters
            \n- python_appears (bool): Whether 'python' exists as a standalone word

    Examples:
        >>> get_str_data("python is great")
        {
            'word_count': 3,
            'longest_word': 'python',
            'python_appears': True
        }
    """
    words = sentence.split()
    longest = max(words, key=lambda x: len(x))
    python_appears = "python" in words

    return {"word_count": len(words), 
            "longest_word": longest,
            "python_appears": python_appears}


if __name__ == "__main__":
    print("test get_str_data")
    print(get_str_data("testing this thing 023 Sdfasdkljnfasdfasdkj python"))