# MediaPipe Tasks API - Combinations and Capabilities

This document outlines the general capabilities of the MediaPipe Tasks API (Python) and the possible combinations of vision tasks that can be run simultaneously or sequentially.

## 1. Core Vision Tasks

These are the primary vision tasks available in the `mediapipe.tasks.python.vision` module.

| Task Name | Description | Key Capabilities | Model File Extension |
| :--- | :--- | :--- | :--- |
| **Face Landmarker** | Detects face landmarks and blendshapes. | • 478 3D landmarks<br>• 52 Blendshapes (expressions)<br>• Transformation Matrix | `.task` |
| **Pose Landmarker** | Detects human body pose. | • 33 3D landmarks<br>• Segmentation mask<br>• Full-body or Upper-body | `.task` |
| **Hand Landmarker** | Detects hand landmarks. | • 21 3D landmarks per hand<br>• Multi-hand support (Left/Right)<br>• Handedness classification | `.task` |
| **Object Detector** | Detects and localizes objects. | • Bounding boxes<br>• Class labels<br>• Confidence scores | `.tflite` |
| **Image Classifier** | Classifies entire images. | • Top-k classification<br>• Confidence scores | `.tflite` |
| **Image Segmenter** | Segments image into regions. | • Category masks<br>• Confidence masks | `.tflite` |
| **Face Detector** | Detects faces (bounding box only). | • Bounding boxes<br>• 6 keypoints (eyes, nose, mouth, ears) | `.tflite` |
| **Interactive Segmenter** | Segments regions based on user input. | • ROI-based segmentation | `.tflite` |

## 2. Common Combinations (Multimodal Vision)

While MediaPipe Tasks are generally independent, they can be combined in a pipeline to create rich applications.

### A. Holistic Tracking (Face + Pose + Hand)
*   **Concept:** Tracking the entire human body, face, and hands simultaneously.
*   **Implementation:** Run `FaceLandmarker`, `PoseLandmarker`, and `HandLandmarker` in parallel (async) or sequence on the same video frame.
*   **Use Cases:**
    *   Full-body avatars (VTubing).
    *   Sign language recognition.
    *   Fitness analysis (Yoga, Gym).
    *   Gesture control.

### B. Face + Hand (Expressive Avatar)
*   **Concept:** Focuses on facial expressions and hand gestures, often used for seated avatars.
*   **Implementation:** `FaceLandmarker` + `HandLandmarker`.
*   **Use Cases:**
    *   Video conferencing filters.
    *   VTubing (upper body).
    *   Emotion analysis with gesture context.

### C. Face + Object Detection (Contextual Analysis)
*   **Concept:** Understanding what the user is looking at or interacting with.
*   **Implementation:** `FaceLandmarker` (gaze estimation via iris) + `ObjectDetector`.
*   **Use Cases:**
    *   Driver monitoring systems (DMS).
    *   Retail analytics (what products are people looking at?).
    *   Accessibility tools.

### D. Pose + Segmentation (Background Removal/Effects)
*   **Concept:** Isolating the user from the background while tracking their movement.
*   **Implementation:** `PoseLandmarker` (with segmentation enabled) or `ImageSegmenter`.
*   **Use Cases:**
    *   Green screen replacement.
    *   Immersive AR games.

## 3. Implementation Modes

MediaPipe Tasks support three running modes, critical for designing your application:

1.  **IMAGE:**
    *   **Input:** Single image.
    *   **Behavior:** Synchronous. Blocks until processing is done.
    *   **Best for:** Batch processing, static image analysis.

2.  **VIDEO:**
    *   **Input:** Video frames with timestamps.
    *   **Behavior:** Synchronous. Context-aware (uses previous frames for smoothing).
    *   **Best for:** Processing pre-recorded video files.

3.  **LIVE_STREAM:**
    *   **Input:** Video frames with timestamps.
    *   **Behavior:** Asynchronous. Returns results via a **callback function**.
    *   **Best for:** Real-time webcam applications, UI threads.
    *   **Note:** Requires careful threading (e.g., using `queue` to pass data to the main thread) to avoid blocking the UI.

## 4. Resource & Model Management

*   **Model Files:** Each task requires a specific model bundle (usually `.task` or `.tflite`). These must be downloaded and provided to the `BaseOptions`.
*   **Delegates:**
    *   **CPU:** Default. Good compatibility.
    *   **GPU:** Faster inference. Requires GPU drivers and compatible hardware.
    *   **TPU (Coral):** Specialized edge hardware acceleration.

## 5. Migration from Legacy (Solutions API)

*   **Legacy:** `mp.solutions.face_mesh`, `mp.solutions.pose`, etc.
*   **New:** `mp.tasks.vision.FaceLandmarker`, `mp.tasks.vision.PoseLandmarker`, etc.
*   **Key Differences:**
    *   The Tasks API is more modular and customizable.
    *   Tasks API uses `.task` bundles which can contain multiple sub-models.
    *   Tasks API has explicit Async support for Live Stream mode.
    *   **Note:** Some legacy visualizers (drawing utils) might not be directly compatible with the raw numpy output of the Tasks API and may require manual rendering (like we did with Ursina).
