# Dataset Restructuring Proposal

## Overview
This document compares the emotion tags used in the synthetic dataset generation (`generate_faces_batch.py`) with the tags found in the filtered real-world dataset (`detect_faces_filter4.json`). The goal is to align the two datasets for combined training.

## Tag Comparison

| Source | Tags |
| :--- | :--- |
| **Old (Synthetic)** | NEUTRAL, HAPPY, SERIOUS, SURPRISED, SAD, ANGRY, DISGUSTED, FEARFUL, CONTEMPTUOUS, BORED, EXCITED, SMUG, CONFUSED, PENSIVE |
| **New (Real-world)** | happy, surprise, sad, angry, disgust, fear |

## Alignment Strategy

To create a consistent classification schema, we propose mapping the synthetic tags to the standard 6 real-world emotion categories (plus 'neutral' if desired as a baseline).

### 1. Direct Mappings (Keep & Rename)
These tags have direct equivalents. We will normalize them to lowercase to match the new dataset.

| Old Tag | New Tag |
| :--- | :--- |
| HAPPY | **happy** |
| SURPRISED | **surprise** |
| SAD | **sad** |
| ANGRY | **angry** |
| DISGUSTED | **disgust** |
| FEARFUL | **fear** |

### 2. Proposed Drops
These tags represent nuanced or compound expressions that do not map cleanly to the basic 6 emotions. Including them might confuse the model or create class imbalance if they are forced into categories they don't fit well.

*   **SERIOUS**: Drop (too ambiguous, overlaps with Neutral/Angry)
*   **CONTEMPTUOUS**: Drop (distinct micro-expression, hard to distinguish from Disgust/Happy mix)
*   **BORED**: Drop (overlaps with Neutral)
*   **SMUG**: Drop (overlaps with Happy/Contempt)
*   **CONFUSED**: Drop (overlaps with Surprise/Angry)
*   **PENSIVE**: Drop (overlaps with Neutral/Sad)

### 3. Special Cases (To Decide)

*   **NEUTRAL**: The real-world dataset (as filtered) does **not** contain "neutral".
    *   *Option A:* Drop NEUTRAL to strictly train on the 6 emotions.
    *   *Option B:* Keep NEUTRAL as a 7th class (recommended for real-world application robustness).
*   **EXCITED**:
    *   *Proposal:* Map to **happy** (high intensity happiness).

## Summary of Actions

1.  **Filter**: When loading the synthetic dataset, exclude samples labeled as SERIOUS, CONTEMPTUOUS, BORED, SMUG, CONFUSED, PENSIVE.
2.  **Map**: Rename the synthetic labels to match the new schema (e.g., FEARFUL -> fear).
3.  **Merge**: Combine `EXCITED` into `happy`.
4.  **Decision**: Decide whether to include `NEUTRAL` as a valid class. (Current recommendation: **Include** for robustness).

## Final Class List
1.  happy (includes HAPPY, EXCITED)
2.  surprise (SURPRISED)
3.  sad (SAD)
4.  angry (ANGRY)
5.  disgust (DISGUSTED)
6.  fear (FEARFUL)
7.  neutral (NEUTRAL) - *Optional/Recommended*
