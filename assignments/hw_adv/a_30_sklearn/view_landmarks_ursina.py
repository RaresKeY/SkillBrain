from ursina import *
import csv
from pathlib import Path

# Initialize Ursina app
app = Ursina()

# Set up camera for 3D navigation
editor_camera = EditorCamera()
editor_camera.position = (0, 0, 0)
editor_camera.target_z = -3

# Text to display current file info
info_text = Text(text="Loading...", position=(-0.85, 0.45), origin=(-0.5, 0.5), scale=1)

# Image preview entity
image_preview = Entity(parent=camera.ui, model='quad', position=(-0.7, -0.35), scale=(0.3, 0.3), visible=False)

# List to hold the current point entities
point_entities = []

# Path to the CSV file
# Assuming script is in assignments/hw_adv/a_30_sklearn/
base_path = Path(__file__).parent
csv_path = base_path / 'dataset' / 'landmarks.csv'

all_rows = []
current_row_index = 0

def load_face(index):
    global point_entities, current_row_index
    
    if not all_rows:
        return

    # Ensure index is within bounds
    current_row_index = index % len(all_rows)
    row = all_rows[current_row_index]
    
    # Cleanup old points
    for p in point_entities:
        destroy(p)
    point_entities.clear()
    
    # Parse row
    # Structure: emotion, filename, face_id, x0, y0, z0, x1, y1, z1, ...
    emotion = row[0]
    filename = row[1]
    face_id = row[2]
    landmarks = row[3:]

    # Load image preview
    try:
        # Construct path relative to project root (assuming standard depth)
        project_root = base_path.parents[2]
        img_path = project_root / filename
        
        if img_path.exists():
            # Try to make it relative to the script directory (asset folder)
            try:
                rel_path = img_path.relative_to(base_path)
                image_preview.texture = str(rel_path)
            except ValueError:
                # If not relative (outside folder), use absolute path
                image_preview.texture = str(img_path)
            
            image_preview.visible = True
        else:
            image_preview.visible = False
            print(f"Image not found: {img_path}")
    except Exception as e:
        print(f"Error loading image: {e}")
        image_preview.visible = False
    
    info_text.text = f"File: {filename}\nEmotion: {emotion}\nFace ID: {face_id}\nIndex: {current_row_index + 1}/{len(all_rows)}\nControls: Left/Right Arrow to switch"

    # Iterate through triplets (x, y, z)
    # MediaPipe Face Mesh has 478 landmarks
    count = 0
    for i in range(0, len(landmarks), 3):
        if i + 2 >= len(landmarks):
            break
            
        try:
            mx = float(landmarks[i])
            my = float(landmarks[i+1])
            mz = float(landmarks[i+2])
            
            # Coordinate transformation for Ursina
            # Data is now in normalized units (face height ~ 1.0), centered at nose, rotated.
            # No extra scaling needed for Ursina default view (units are meters).
            x = mx
            y = my
            z = mz
            
            # Create point
            # Color points to make it look nicer (e.g., gradient)
            # Landmarks 0-468 are the mesh, 468-478 are irises
            c = color.azure
            if count >= 468:
                c = color.red # Irises
                
            e = Entity(model='sphere', color=c, scale=0.015, position=(x, y, z))
            point_entities.append(e)
            count += 1
            
        except ValueError:
            continue

def input(key):
    if key == 'right arrow':
        load_face(current_row_index + 1)
    elif key == 'left arrow':
        load_face(current_row_index - 1)
    elif key == 'escape':
        application.quit()

# Read CSV Data
if csv_path.exists():
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        try:
            header = next(reader) # Skip header
            all_rows = list(reader)
        except StopIteration:
            pass

    if all_rows:
        print(f"Loaded {len(all_rows)} faces.")
        load_face(0)
    else:
        info_text.text = "Error: No data found in landmarks.csv"
else:
    info_text.text = f"Error: File not found at\n{csv_path}"

app.run()
