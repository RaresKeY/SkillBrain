"""Config for PySide6 YouTube feed with YOLO overlay."""

# Stream source (YouTube live/video URL)
SOURCE_URL = 'https://www.youtube.com/channel/UC6AlfoRUeH4B1an_R5YSSTw/live'

# Recommended model for this wide outdoor scene (good small-object recall vs speed)
MODEL_NAME = 'yolo11m.pt'

# Available model options shown in the UI
MODEL_OPTIONS = ['yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt', 'yolo11m.pt', 'yolo11l.pt']

# Detection and inference settings
CONF_THRESHOLD = 0.3
INFERENCE_IMAGE_SIZE = 960
INFER_EVERY_N_FRAMES = 2

# Draw only selected COCO classes (empty list means all classes)
DRAW_CLASSES = ['person', 'boat', 'bird', 'bench', 'dog', 'cat', 'bicycle', 'motorcycle', 'car', 'bus', 'truck']

# yt-dlp stream format selector
YDL_FORMAT = 'best[height<=1080][protocol^=m3u8]/best[height<=1080]/best'

# Save detection snapshots
SAVE_DETECTIONS = True
SAVE_DIR = 'detections'
SAVE_MIN_INTERVAL_SECONDS = 1.0

# UI defaults
WINDOW_TITLE = 'YouTube Feed + YOLO Overlay (PySide6)'
WINDOW_WIDTH = 1360
WINDOW_HEIGHT = 840
