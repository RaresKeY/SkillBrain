# Image Generation V2 Structure & Mapping

## Overview
The V2 generation script (`generate_faces_batch_v2.py`) aligns synthetic data generation with the standardized 7-class emotion model used in the training pipeline. It also introduces minimized metadata codes for cleaner file management.

## Minimized Metadata Codes
To reduce filename length while preserving attributes, the following codes are used in filenames:

| Attribute | Code | Meaning |
| :--- | :--- | :--- |
| **Sex** | M, F | Male, Female |
| **Age** | CHD, TEN, YNG, MID, ELD | Child, Teenager, Young Adult, Middle Aged, Elderly |
| **Ethnicity** | CAU, AFR, ASN, LAT, IND, MEA | Caucasian, African, Asian, Latino, Indian, Middle Eastern |
| **Attractiveness** | AVG, ATT, MOD, UGL | Average, Attractive, Model, Ugly |

## Standardized Emotion Mapping
The generation script uses the following mapping logic (matching `train_model_combined.py`) to ensure consistency between filenames and dataset labels.

| Prompt Expression | Filename Tag | Target Label (Training) |
| :--- | :--- | :--- |
| Neutral expression | `neutral` | neutral |
| Serious gaze | `serious` | neutral |
| Bored / Indifferent | `bored` | neutral |
| Smiling / Happy | `happy` | happy |
| Surprised | `surprised` | surprise |
| Melancholic / Sad | `sad` | sad |
| Angry | `angry` | angry |
| Disgusted | `disgusted` | disgust |
| Fearful | `fearful` | fear |

### Dropped Expressions
The following expressions from V1 are excluded from V2 to maintain alignment with the real-world filtered dataset:
*   Excited (Mapped to happy in proposal, but dropped in V2 for strictness)
*   Contemptuous
*   Smug
*   Confused
*   Pensive

## Filename Structure
The new filename format is:
`[SEX]_[AGE]_[ETHNICITY]_[EXPRESSION]_[ATTRACTIVENESS].png`

**Example:**
`m_yng_cau_serious_avg.png`
*   **M**: Male
*   **YNG**: Young Adult
*   **CAU**: Caucasian
*   **serious**: Expression (Mapped to 'neutral' in training)
*   **AVG**: Average Attractiveness

## Key Changes in Script
1.  **Pydantic Titles**: Added minimized titles (`title="S"`, etc.) to `FaceGenerationPrompt` fields.
2.  **Standardized Enums**: Shortened Enum member names for metadata.
3.  **V2 Directory**: Outputs to `dataset/generated_faces_v2` to avoid mixing with legacy data.
4.  **Implicit Mapping Support**: By using tags like `serious` and `bored`, the script generates diverse visuals that the training pipeline automatically consolidates into the `neutral` class.
