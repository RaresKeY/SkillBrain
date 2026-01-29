from helpers.folder_tools import file_generator, ft
from pathlib import Path
import json
import cv2

folder_path = Path("/media/mintmainog/c21d735b-a894-4487-8dc4-b83f31f0a84c/people_datasets/EmoSet-118K/annotation")
SCRIPT_FOLDER = Path(__file__).resolve().parent
print(SCRIPT_FOLDER)

files = file_generator(folder_path, ft.JSON, whitelist=["/media"])
face_detects_prelim = dict()

for file in files:
    fpo = Path(file)
    with open(fpo, 'r', encoding='utf-8') as f:
        data: dict = json.load(f)

    if "facial_expression" in data.keys():
        face_detects_prelim[str(fpo)] = data["facial_expression"]

with open(SCRIPT_FOLDER / "detect_faces_prelim.json", 'w', encoding='utf-8') as f:
    json.dump(face_detects_prelim, f, indent=2)