# OpenCV Surveillance & Fire Safety System

## Project Structure

```text
opencv_surveillance_project/
├── surveillance_system.py
├── test_results/
│   ├── .gitkeep
│   └── event_log.txt (generated at runtime)
├── README.md
└── reflection_answers.md
```

## How to Run

1. Create and activate the project virtual environment (repo-level):

```bash
# From repository root:
python -m venv .venv
source .venv/bin/activate
pip install -r assignments/hw_adv/a_28_opencv/requirements.txt
```

2. Run the surveillance system:

```bash
cd assignments/hw_adv/a_28_opencv/opencv_surveillance_project
# Webcam mode (default)
python surveillance_system.py --show-debug

# Pre-recorded video mode
python surveillance_system.py --video videos/test_video.mp4 --show-debug
```

3. Quit with `q` in the camera window.

## Run Warmup Exercises

```bash
cd assignments/hw_adv/a_28_opencv/opencv_surveillance_project
python exercises/exercise_1_blue_detector.py --image data/colorful.jpg
python exercises/exercise_2_text_enhancement.py --image data/old_text.jpg
python exercises/exercise_3_contour_analysis.py --image data/objects.jpg
```

## Prepared Exercise Data

- `data/colorful.jpg` (blue object sample)
  - Source: `https://commons.wikimedia.org/wiki/Special:FilePath/Cobalt%20Blue%20Glass%20Ball%2001.jpg`
- `data/old_text.jpg` (old manuscript text sample)
  - Source: `https://commons.wikimedia.org/wiki/Special:FilePath/Page%20(Manuscript).jpg`
- `data/objects.jpg` (multi-object sample)
  - Source: `https://raw.githubusercontent.com/opencv/opencv/master/samples/data/fruits.jpg`

## Implemented Features

- Live webcam stream
- Motion detection using frame differencing + contour filtering
- Smoke-like region detection using HSV + morphological cleanup
- Growth-aware smoke alert using rolling smoke history
- Visual alarms with highlighted frames and status text
- Timestamped console logging and file logging (`test_results/event_log.txt`)
- Event snapshots saved to `test_results/` as:
  - `motion_YYYY-MM-DD_HH-MM-SS.jpg`
  - `smoke_YYYY-MM-DD_HH-MM-SS.jpg`

## Threshold Values Used

- `motion_threshold = 2000`
  - Triggers alert when total moving contour area is significant.
- `min_contour_area = 1000`
  - Filters tiny movements/noise.
- `smoke_pixel_threshold = 5000`
  - Triggers when a substantial low-saturation bright region appears.
- Smoke HSV range:
  - Lower: `[0, 0, 120]`
  - Upper: `[180, 65, 255]`
  - Rationale: captures gray/white low-saturation bright regions.
- `smoke_growth_factor = 1.35`
  - Also alerts when recent smoke area average grows by 35%+ vs earlier frames.

## Testing Methodology

### Motion detection tests

1. Keep scene static for baseline.
2. Move hand through camera view.
3. Walk across frame.
4. Verify moving boxes are drawn and motion alerts are logged.

### Smoke detection tests (safe simulation)

1. Move white/gray paper upward slowly.
2. Use steam from boiled water at safe distance.
3. Verify smoke pixel count rises and smoke alert is triggered.

## Challenges and Solutions

- Lighting changes can mimic motion/noise.
  - Solution: Gaussian blur, minimum contour area, and threshold tuning.
- Bright objects can resemble smoke.
  - Solution: combine pixel threshold with growth-over-time check.
- Event spam can produce too many snapshots.
  - Solution: one snapshot per alert type per second guard.

## Optional Demo Video

- Add a link here if you record a demo.
