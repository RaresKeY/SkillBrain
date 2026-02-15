import os
import cv2
import numpy as np
from deepface import DeepFace
import dlib
from pathlib import Path

# Configuration
# Set to -1 to use CPU, or 0 for GPU if available and configured
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Paths
current_dir = Path(__file__).parent.absolute()
os.chdir(current_dir)

# Dlib Setup
DLIB_MODEL_PATH = "shape_predictor_68_face_landmarks.dat"
if not os.path.exists(DLIB_MODEL_PATH):
    print(f"Error: Dlib model not found at {DLIB_MODEL_PATH}. Please download it from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
    exit(1)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(DLIB_MODEL_PATH)

def draw_dlib_landmarks(img):
    """
    Detects face and draws 68 landmarks on the image.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)

    for rect in rects:
        # Draw bounding box
        x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Get landmarks
        shape = predictor(gray, rect)

        # Draw landmarks
        for i in range(0, 68):
            p = shape.part(i)
            cv2.circle(img, (p.x, p.y), 2, (0, 0, 255), -1)

    return img

def verify_and_visualize(source_img_path, target_img_path, output_name):
    print(f"Processing: {source_img_path} vs {target_img_path}")

    try:
        # 1. DeepFace Verification
        result = DeepFace.verify(
            img1_path=source_img_path,
            img2_path=target_img_path,
            model_name="ArcFace",
            enforce_detection=False # Don't crash if deepface misses a face, we handle it
        )
        print(f"DeepFace Result: {result['verified']} (Distance: {result['distance']:.4f})")

        # 2. Load Images for Visualization
        img1 = cv2.imread(source_img_path)
        img2 = cv2.imread(target_img_path)

        if img1 is None or img2 is None:
            print("Error loading images.")
            return

        # 3. Draw Dlib Landmarks
        img1 = draw_dlib_landmarks(img1)
        img2 = draw_dlib_landmarks(img2)

        # 4. Resize for concatenation
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        if h1 != h2:
            scale = h1 / h2
            new_w2 = int(w2 * scale)
            img2 = cv2.resize(img2, (new_w2, h1))

        # 5. Concatenate
        combined = np.hstack((img1, img2))

        # 6. Add Status Text
        is_verified = result.get('verified', False)
        distance = result.get('distance', 0)

        text = f"Verified: {is_verified} (Dist: {distance:.4f})"
        color = (0, 255, 0) if is_verified else (0, 0, 255)

        # Text Background
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        text_w, text_h = text_size
        cv2.rectangle(combined, (10, 10), (10 + text_w + 20, 10 + text_h + 20), (0, 0, 0), -1)
        cv2.putText(combined, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # 7. Save Output
        output_path = f"result_{output_name}.jpg"
        cv2.imwrite(output_path, combined)
        print(f"Saved: {output_path}\n")

    except Exception as e:
        print(f"An error occurred: {e}")

def main():
    # Define Source
    img_source = "media/obama.jpg"

    # Define Target (File or Folder)
    # You can change this to a single file path or a folder path
    check_target = "media" 

    if not os.path.exists(img_source):
        print(f"Source image {img_source} not found.")
        return

    if os.path.isfile(check_target):
        # Single file comparison
        verify_and_visualize(img_source, check_target, "single_check")

    elif os.path.isdir(check_target):
        # Directory comparison
        files = os.listdir(check_target)
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')

        count = 0
        for f in files:
            if f.lower().endswith(valid_extensions):
                full_path = os.path.join(check_target, f)

                # Skip comparing with itself
                if os.path.abspath(full_path) == os.path.abspath(img_source):
                    continue

                verify_and_visualize(img_source, full_path, f"check_{count}")
                count += 1
    else:
        print(f"Target {check_target} not found.")

if __name__ == "__main__":
    main()
