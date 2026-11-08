# Smart Webcam Assistant

Made by **Al3rab**

A single-camera Python project that runs entirely on your laptop webcam:

- **Entry/Exit logging** — logs ENTRY/EXIT to `webcam_log.csv` with timestamp
- **Drowsiness detection** — EAR-based detection + 5-second sound alarm after eyes closed 3 seconds
- **Full skeletons** (visual only, no control) — advanced hand (21, up to 2 hands) + body pose (33, no face) + object boxes (e.g. cup)
- **Distance estimate** — shows face distance (Close/Good/Far/Too far) in real time
- **Sleek HUD** — top status bar with EAR / Distance / In-Out / FPS, bottom hint bar
- **Efficient** — face+hand every frame, pose/object every 3rd frame

Camera only, no media/gesture control. Face/eye detection runs internally (no face drawing).

## Project structure

```
Smart-Webcam-Assistant/
├── smart_webcam.py      # CLI version (OpenCV window + terminal)
├── gui_app.py           # Camera-only GUI (window shows ONLY the camera)
├── requirements.txt
├── run.sh               # Launch CLI
├── run_gui.sh           # Launch GUI
├── README.md
└── LICENSE
```

Models (`*.task`, `*.tflite`) download automatically on first run — they are not committed (see `.gitignore`).

## Install

```bash
python3 -m venv venv
source venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Requires Python 3.10+. First run downloads MediaPipe models (internet required once).

## Run

```bash
# GUI (camera-only window, 960x720)
./run_gui.sh
# or
source venv/bin/activate
python gui_app.py

# CLI
./run.sh
# or
python smart_webcam.py
```

Close with `Q`, `Esc`, or the window `X` button. Headless: `env -u DISPLAY python smart_webcam.py` prints events to the terminal.

## Tune

At the top of `smart_webcam.py`:

- `EAR_THRESHOLD = 0.25` — drowsiness threshold
- `DROWSINESS_ALARM_SEC = 3.0` — alarm after eyes closed this long (5s long sound)
- `EXIT_THRESHOLD_SEC = 5` — seconds with no face → EXIT

## Gestures / skeletons

- Hand: green glow skeleton, palm fill, bounding box, fingertip labels (Thumb/Index/Middle/Ring/Pinky), up to 2 hands
- Pose: magenta body skeleton (face points excluded)
- Objects: yellow bounding boxes with labels, e.g. `cup 0.95`

## License

MIT — see `LICENSE`.
