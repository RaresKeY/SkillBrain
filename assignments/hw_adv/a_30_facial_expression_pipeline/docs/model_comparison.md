# Model Comparison: Gemini 3 Flash vs. Gemma 3 27B

## Statistical Accuracy (Top 80 Images)
Based on ground truth extracted from filenames (mapping 'smug/excited' -> happy, 'serious/pensive' -> neutral, etc.)

| Metric | Gemini 3 Flash | Gemma 3 27B |
| :--- | :--- | :--- |
| **Overall Accuracy** | **76.25%** | **76.25%** |

### Accuracy per Emotion
| Emotion | Count | Gemini % | Gemma % |
| :--- | :--- | :--- | :--- |
| angry | 11 | 72.7% | 72.7% |
| disgust | 4 | 75.0% | 25.0% |
| fear | 14 | 71.4% | 78.6% |
| happy | 13 | 76.9% | 92.3% |
| neutral | 17 | 82.4% | 70.6% |
| sad | 9 | 88.9% | 88.9% |
| surprise | 12 | 66.7% | 75.0% |

## Examples of Disagreements

### Case 1: Subtlety (Neutral vs Sad)
*   **File:** `male_teenager_latino_neutral_average.png`
*   **Gemini:** `neutral` (Correct)
*   **Gemma:** `sad` (Incorrect - Over-sensitive to resting facial features)

### Case 2: Intensity (Fear vs Angry)
*   **File:** `male_middle_caucasian_fearful_average.png`
*   **Gemini:** `fear` (Correct)
*   **Gemma:** `angry` (Incorrect - Likely misinterpreted furrowed brows)

### Case 3: Nuance (Disgust vs Angry)
*   **File:** `male_young_caucasian_disgusted_average.png`
*   **Gemini:** `disgust` (Correct)
*   **Gemma:** `angry` (Incorrect)
