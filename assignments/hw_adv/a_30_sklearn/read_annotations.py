from helpers.folder_tools import file_generator, ft
from pathlib import Path
import json
import cv2

def _load_local_paths() -> dict:
    here = Path(__file__).resolve()
    for base in [here.parent, *here.parents]:
        cfg = base / ".local_paths.json"
        if cfg.exists():
            try:
                return json.loads(cfg.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return {}
    return {}


_LOCAL_PATHS = _load_local_paths()
folder_path = Path(_LOCAL_PATHS.get("emotions_annotation_folder", "data/annotations"))
SCRIPT_FOLDER = Path(__file__).resolve().parent
print(SCRIPT_FOLDER)

files = file_generator(folder_path, ft.JSON, whitelist=[str(folder_path)])
face_detects_prelim = dict()

for file in files:
    fpo = Path(file)
    with open(fpo, 'r', encoding='utf-8') as f:
        data: dict = json.load(f)

    if "facial_expression" in data.keys():
        face_detects_prelim[str(fpo)] = data["facial_expression"]

with open(SCRIPT_FOLDER / "detect_faces_prelim.json", 'w', encoding='utf-8') as f:
    json.dump(face_detects_prelim, f, indent=2)
