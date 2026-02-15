import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import cv2
import numpy as np
from deepface import DeepFace
from pathlib import Path

script_dir = Path(__file__).parent.absolute()
os.chdir(script_dir)

img1_path = "media/obama.jpg"
img2_path = "media/obama2.jpg"

print(f"Processing {img1_path} and {img2_path}...")
result = DeepFace.verify(img1_path=img1_path, img2_path=img2_path, model_name="ArcFace")
print(result)

# Visualization
if result:
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    if img1 is None or img2 is None:
        print("Error loading images for visualization.")
    else:
        # Draw for img1
        if 'facial_areas' in result and 'img1' in result['facial_areas']:
            area1 = result['facial_areas']['img1']
            x1, y1, w1, h1 = area1['x'], area1['y'], area1['w'], area1['h']
            cv2.rectangle(img1, (x1, y1), (x1+w1, y1+h1), (0, 255, 0), 2)

        # Draw for img2 (before resize)
        if 'facial_areas' in result and 'img2' in result['facial_areas']:
            area2 = result['facial_areas']['img2']
            x2, y2, w2, h2 = area2['x'], area2['y'], area2['w'], area2['h']
            cv2.rectangle(img2, (x2, y2), (x2+w2, y2+h2), (0, 255, 0), 2)

        # Resize img2 to match img1 height for clean concatenation
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        if h1 != h2:
            scale = h1 / h2
            new_w2 = int(w2 * scale)
            img2 = cv2.resize(img2, (new_w2, h1))

        # Concatenate
        combined = np.hstack((img1, img2))

        # Add text
        is_verified = result.get('verified', False)
        distance = result.get('distance', 0)
        
        text = f"Verified: {is_verified} (Dist: {distance:.4f})"
        color = (0, 255, 0) if is_verified else (0, 0, 255)

        # Add a black background for text for better visibility
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        text_w, text_h = text_size
        cv2.rectangle(combined, (10, 10), (10 + text_w + 20, 10 + text_h + 20), (0, 0, 0), -1)
        cv2.putText(combined, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        output_path = "verification_result.jpg"
        cv2.imwrite(output_path, combined)
        print(f"Result image saved to {os.path.abspath(output_path)}")