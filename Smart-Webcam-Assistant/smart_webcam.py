"""
Smart Webcam Assistant - CAMERA ONLY
=====================================
Camera-only project (no media/gesture control). Features:
  1. Entry/Exit Logger (logs to webcam_log.csv)
  2. Drowsiness Detection (via EAR) with 3-second sound alarm (5s long alarm)
  3. Skeletons (visual only): hand (21) • pose (33) • objects (cup...) - face/eye detection without drawing

Important about MediaPipe:
  - New versions (1.x) removed the old mp.solutions API.
  - This code uses the modern Tasks API: FaceLandmarker + HandLandmarker.
  - On first run it auto-downloads the model files, so internet is needed once.

Run:
  - With a display (DISPLAY set): OpenCV window + terminal prints.
  - Without a display (no DISPLAY): no window, prints events to terminal.

Requirements:
  - Python 3.10+
  - opencv-python, mediapipe (>=0.10.30), numpy
"""

# ---------------- Core imports ----------------
import csv
import datetime
import os
import subprocess
import threading
import time
import urllib.request
import wave

import cv2
import mediapipe as mp
import numpy as np

# ---------------- Configurable settings ----------------
# >> Tune these values to your liking <<
EAR_THRESHOLD = 0.25          # Drowsiness threshold: EAR below this means eye is closed
EAR_CONSEC_FRAMES = 20        # Consecutive frames below threshold to confirm drowsiness
DROWSINESS_ALARM_SEC = 3.0    # Sound alarm after eyes stay closed this many seconds
ALARM_ENABLED = True          # Drowsiness sound alarm (always on)
ALARM_REPEAT_SEC = 5.5        # Repeat the alarm every this many seconds while still drowsy
ALARM_BEEP_HZ = 1200          # Alarm tone frequency (5s long tone)
EXIT_THRESHOLD_SEC = 5        # Seconds with no face before logging EXIT
LOG_FILE = "webcam_log.csv"   # Entry/exit log file
CAMERA_INDEX = 0              # 0 = built-in laptop camera

# MediaPipe Tasks model files (auto-downloaded if missing)
FACE_MODEL = "face_landmarker.task"
HAND_MODEL = "hand_landmarker.task"
POSE_MODEL = "pose_landmarker.task"
OBJECT_MODEL = "object_detector.tflite"
FACE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
OBJECT_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite"

# Eye landmarks in the Face Landmarker model (MediaPipe)
# Order matters for EAR: [outer corner, top 1, top 2, inner corner, bottom 2, bottom 1]
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Hand skeleton connections (21 landmarks) - used to draw an accurate skeleton
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (0, 9), (9, 10), (10, 11), (11, 12),      # middle
    (0, 13), (13, 14), (14, 15), (15, 16),   # ring
    (0, 17), (17, 18), (18, 19), (19, 20),   # pinky
]
# Body pose connections (33 landmarks)
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (17, 19), (15, 21),
    (12, 14), (14, 16), (16, 18), (18, 20), (14, 22),
    (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (27, 29),
    (24, 26), (26, 28), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
]

# Is there a graphical display?
HAS_DISPLAY = bool(os.environ.get("DISPLAY"))


# ---------------- Helper functions ----------------
def timestamp_now():
    """Return the current date and time as text."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def print_event(message):
    """Print a message to the terminal with the time and date."""
    print(f"[{timestamp_now()}] {message}", flush=True)


def log_event(event_type):
    """Log an entry/exit event to CSV and print a terminal message."""
    now = timestamp_now()
    # Create the file with headers on first run
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["type", "timestamp"])
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([event_type, now])
    if event_type == "ENTRY":
        print_event("Entry logged: face detected for the first time")
    else:
        print_event(f"Exit logged: face missing for {EXIT_THRESHOLD_SEC} seconds")


def ensure_models():
    """Download MediaPipe model files if they are missing (first run only)."""
    for path, url in [(FACE_MODEL, FACE_MODEL_URL), (HAND_MODEL, HAND_MODEL_URL),
                      (POSE_MODEL, POSE_MODEL_URL), (OBJECT_MODEL, OBJECT_MODEL_URL)]:
        if not os.path.exists(path):
            print_event(f"Downloading MediaPipe model: {path} (first run only)...")
            try:
                urllib.request.urlretrieve(url, path)
                print_event(f"Model downloaded successfully: {path}")
            except Exception as exc:
                print_event(f"Failed to download model {path}: {exc}")
                return False
    return True


def eye_aspect_ratio(landmarks, eye_indices, img_w, img_h):
    """Compute the Eye Aspect Ratio (EAR) to tell if an eye is open or closed."""
    pts = []
    for idx in eye_indices:
        lm = landmarks[idx]
        pts.append((int(lm.x * img_w), int(lm.y * img_h)))
    # A and B are vertical distances between eyelids, C is horizontal corner distance
    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    if C == 0:
        return 0.0
    return (A + B) / (2.0 * C)


def estimate_distance_cm(landmarks, img_w):
    """
    Estimate face-to-camera distance (cm) from face width.
    Uses real face width ~16cm and focal ~0.86*img_w (approx for 60deg FOV).
    Returns 0 if too small.
    """
    if not landmarks:
        return 0.0
    xs = [lm.x for lm in landmarks]
    face_w_norm = max(xs) - min(xs)
    if face_w_norm <= 0.01:
        return 0.0
    face_w_px = face_w_norm * img_w
    focal = img_w * 0.86
    return (16.0 * focal) / face_w_px


def distance_label(dist_cm):
    """Return a short label for the distance."""
    if dist_cm <= 0:
        return "Unknown"
    if dist_cm < 40:
        return "Close"
    if dist_cm < 80:
        return "Good"
    if dist_cm < 120:
        return "Far"
    return "Too far"


HAND_FINGERTIP_LABELS = {4: "Thumb", 8: "Index", 12: "Middle", 16: "Ring", 20: "Pinky"}


def draw_hand_skeleton(frame, landmarks, img_w, img_h, smoothed_pts=None):
    """
    Advanced hand skeleton: palm fill, bounding box, thick glow lines,
    bordered joints, fingertip name labels.
    """
    def to_px(pt):
        return (int(pt.x * img_w), int(pt.y * img_h))
    pts = smoothed_pts if smoothed_pts is not None else [to_px(lm) for lm in landmarks]
    pts = [(int(x), int(y)) for x, y in pts]

    # 1) Palm fill (wrist + MCPs) semi-transparent
    palm_idx = [0, 5, 9, 13, 17]
    palm_pts = np.array([pts[i] for i in palm_idx], dtype=np.int32)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [palm_pts], (0, 255, 0))
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
    cv2.polylines(frame, [palm_pts], True, (0, 255, 0), 2)

    # 2) Bounding box around the hand
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    x1, y1 = max(0, min(xs) - 12), max(0, min(ys) - 12)
    x2, y2 = min(img_w - 1, max(xs) + 12), min(img_h - 1, max(ys) + 12)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
    cv2.putText(frame, "Hand", (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    # 3) Connections: thick dark under + bright over (glow)
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 180, 0), 5)
        cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)

    # 4) Joints with white border (strong, easy to see)
    FINGERTIPS = {4, 8, 12, 16, 20}
    for i, c in enumerate(pts):
        if i == 0:
            # Wrist: big white dot with black border
            cv2.circle(frame, c, 8, (0, 0, 0), -1)
            cv2.circle(frame, c, 6, (255, 255, 255), -1)
        elif i in FINGERTIPS:
            cv2.circle(frame, c, 7, (255, 255, 255), -1)
            cv2.circle(frame, c, 5, (255, 255, 0), -1)
            label = HAND_FINGERTIP_LABELS.get(i, "")
            cv2.putText(frame, label, (c[0] + 8, c[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        else:
            cv2.circle(frame, c, 5, (255, 255, 255), -1)
            cv2.circle(frame, c, 3, (0, 0, 255), -1)


def smooth_points(prev, curr, alpha=0.65):
    """Exponential smoothing for stable, jitter-free skeleton."""
    if prev is None:
        return curr
    return [(alpha * c[0] + (1 - alpha) * p[0], alpha * c[1] + (1 - alpha) * p[1])
            for c, p in zip(curr, prev)]


def draw_hud(frame, ear, dist, dist_lbl, entry_text, fps):
    """Draw a sleek HUD bar (top) + hint bar (bottom)."""
    h, w, _ = frame.shape
    bar = frame.copy()
    cv2.rectangle(bar, (0, 0), (w, 46), (0, 0, 0), -1)
    cv2.addWeighted(bar, 0.55, frame, 0.45, 0, frame)
    cv2.line(frame, (0, 46), (w, 46), (0, 255, 0), 2)
    ear_s = f"{ear:.2f}" if ear is not None else "--"
    dist_s = f"{dist:.0f}cm" if dist and dist > 0 else "--"
    line1 = f"EAR {ear_s}   Dist {dist_s} ({dist_lbl})   {entry_text}   {fps:.0f} FPS"
    dist_col = (0, 255, 0) if dist_lbl == "Good" else (0, 255, 255) if dist_lbl in ("Close", "Far") else (255, 255, 255)
    cv2.putText(frame, line1, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, dist_col, 2)
    bar2 = frame.copy()
    cv2.rectangle(bar2, (0, h - 36), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(bar2, 0.45, frame, 0.55, 0, frame)
    cv2.line(frame, (0, h - 36), (w, h - 36), (255, 255, 255), 1)
    cv2.putText(frame, "Q / Esc close  -  eyes closed 3s = 5s alarm", (12, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


def draw_face_mesh(frame, landmarks, img_w, img_h):
    """Draw the predicted face oval (convex hull) + eye highlights. No dense dots."""
    def to_px(pt):
        return (int(pt.x * img_w), int(pt.y * img_h))
    pts = [to_px(lm) for lm in landmarks]
    if len(pts) >= 4:
        hull = cv2.convexHull(np.array(pts, dtype=np.int32))
        cv2.polylines(frame, [hull], True, (255, 0, 0), 2)
    # Eye highlights: circles around the eye regions (cyan)
    for eye in [LEFT_EYE, RIGHT_EYE]:
        eye_pts = [to_px(landmarks[i]) for i in eye]
        if eye_pts:
            cx = int(sum(p[0] for p in eye_pts) / len(eye_pts))
            cy = int(sum(p[1] for p in eye_pts) / len(eye_pts))
            cv2.circle(frame, (cx, cy), 14, (255, 255, 0), 2)


def draw_pose_skeleton(frame, landmarks, img_w, img_h):
    """Draw the body pose skeleton (magenta), excluding face points (0-10)."""
    def to_px(pt):
        return (int(pt.x * img_w), int(pt.y * img_h))
    FACE_INDICES = set(range(0, 11))  # 0-10 are face/nose/eyes - never draw these
    for a, b in POSE_CONNECTIONS:
        if a in FACE_INDICES or b in FACE_INDICES:
            continue
        if a < len(landmarks) and b < len(landmarks):
            cv2.line(frame, to_px(landmarks[a]), to_px(landmarks[b]), (255, 0, 255), 2)
    for i, lm in enumerate(landmarks):
        if i in FACE_INDICES:
            continue
        cv2.circle(frame, to_px(lm), 3, (255, 0, 255), -1)


def draw_objects(frame, detections):
    """Draw precise bounding boxes + labels (e.g. cup) for each object."""
    for det in detections:
        try:
            box = det.bounding_box
            cat = det.categories[0] if det.categories else None
            label = cat.category_name if cat else "object"
            score = cat.score if cat else 0.0
            x = int(box.origin_x); y = int(box.origin_y)
            bw = int(box.width); bh = int(box.height)
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 255), 2)
            cv2.putText(frame, f"{label} {score:.2f}", (x, max(0, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        except Exception:
            continue


def _beep_sync(freq=1200, duration=5.0):
    """Play a loud LONG alarm tone (blocking). Never raises."""
    try:
        import numpy as _np
        sr = 44100
        # One long continuous loud tone for `duration` seconds (default 2.5s)
        t = _np.linspace(0, duration, int(sr * duration), endpoint=False)
        samples = (_np.sin(2 * _np.pi * freq * t) * 0.9 * 32767).astype(_np.int16)
        wav_path = f"/tmp/drowsy_alarm_{os.getuid()}.wav"
        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
            wf.writeframes(samples.tobytes())
        # Try every available player (first success wins)
        for cmd in (["paplay", wav_path], ["aplay", wav_path],
                    ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", wav_path],
                    ["mpv", "--no-video", wav_path]):
            try:
                r = subprocess.run(cmd, timeout=6,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if r.returncode == 0:
                    return True
            except Exception:
                continue
    except Exception:
        pass
    # Method 2: 'beep' command if installed
    try:
        r = subprocess.run(["beep", "-f", str(freq), "-l", str(int(duration * 1000))],
                           timeout=3, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if r.returncode == 0:
            return True
    except Exception:
        pass
    # Method 3: terminal BEL character (always works, may be silent without audio)
    try:
        import sys
        print("\a", file=sys.stderr, end="", flush=True)
        return True
    except Exception:
        return False


def play_alarm_sound(freq=1200, duration=5.0):
    """Play the drowsiness alarm sound in a background thread (non-blocking)."""
    def _run():
        _beep_sync(freq=freq, duration=duration)
    threading.Thread(target=_run, daemon=True).start()


# ---------------- Main function ----------------
def main():
    print_event("Smart Webcam Assistant started")
    if not HAS_DISPLAY:
        print_event("No graphical display (DISPLAY is empty). Text mode: printing events to terminal only")

    # Download model files if needed
    if not ensure_models():
        print_event("Error: could not download MediaPipe models. Check your internet connection and try again")
        return

    # Import the modern Tasks API (old mp.solutions was removed in 1.x)
    from mediapipe.tasks.python.core.base_options import BaseOptions
    from mediapipe.tasks.python.vision import (
        FaceLandmarker,
        FaceLandmarkerOptions,
        HandLandmarker,
        HandLandmarkerOptions,
        PoseLandmarker,
        PoseLandmarkerOptions,
        ObjectDetector,
        ObjectDetectorOptions,
    )
    from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

    # Create FaceLandmarker and HandLandmarker
    face_options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=FACE_MODEL),
        running_mode=VisionTaskRunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    hand_options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HAND_MODEL),
        running_mode=VisionTaskRunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7,
    )
    pose_options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=POSE_MODEL),
        running_mode=VisionTaskRunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    object_options = ObjectDetectorOptions(
        base_options=BaseOptions(model_asset_path=OBJECT_MODEL),
        running_mode=VisionTaskRunningMode.IMAGE,
        score_threshold=0.4,
        max_results=5,
    )
    face_landmarker = FaceLandmarker.create_from_options(face_options)
    hand_landmarker = HandLandmarker.create_from_options(hand_options)
    try:
        pose_landmarker = PoseLandmarker.create_from_options(pose_options)
    except Exception as exc:
        print_event(f"Warning: pose detector unavailable ({exc}) - continuing without it")
        pose_landmarker = None
    try:
        object_detector = ObjectDetector.create_from_options(object_options)
    except Exception as exc:
        print_event(f"Warning: object detector unavailable ({exc}) - continuing without it")
        object_detector = None

    # Check the camera
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print_event("Error: camera not found. Check the connection and camera permissions")
        face_landmarker.close()
        hand_landmarker.close()
        if pose_landmarker:
            pose_landmarker.close()
        if object_detector:
            object_detector.close()
        return

    # State variables
    face_detected = False          # Is a face visible right now?
    last_seen_time = time.time()   # Last time the face was seen
    counter_drowsy = 0             # Consecutive drowsy frames counter
    drowsy_alerted = False         # Has the drowsiness alert already fired?
    eye_closed_since = None        # Timestamp when eyes first closed (for 3s alarm)
    eyes_open_since = None         # Grace timer (ignore brief EAR spikes)
    last_alarm_time = 0            # Last alarm sound time (throttles repeats)
    hand_smooths = [None, None]    # Smoothed points per hand (up to 2 hands)
    frame_idx = 0                  # Frame number (throttles EAR prints in text mode)
    fps = 0.0                      # Smoothed FPS for the HUD
    _last_t = time.time()          # Last frame time for FPS
    _cached_pose = None            # Cached pose result (efficiency)
    _cached_obj = None             # Cached object result (efficiency)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print_event("Warning: could not read a frame from the camera")
                time.sleep(0.1)
                continue

            # Mirror the frame for a more natural view
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = np.ascontiguousarray(rgb)

            # Convert to a MediaPipe Tasks image and detect (efficient: face+hand every frame, pose/object every 3rd)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            face_result = face_landmarker.detect(mp_image)
            hand_result = hand_landmarker.detect(mp_image)
            frame_idx += 1
            if frame_idx % 3 == 0:
                if pose_landmarker:
                    _cached_pose = pose_landmarker.detect(mp_image)
                if object_detector:
                    _cached_obj = object_detector.detect(mp_image)
            pose_result = _cached_pose
            object_result = _cached_obj
            current_time = time.time()
            if _last_t:
                _dt = max(0.001, current_time - _last_t)
                fps = 0.9 * fps + 0.1 * (1.0 / _dt) if fps else (1.0 / _dt)
            _last_t = current_time

            # ---- Part 1: Entry/Exit logging ----
            face_present = bool(face_result.face_landmarks)
            if face_present:
                if not face_detected:
                    face_detected = True
                    log_event("ENTRY")
                    if HAS_DISPLAY:
                        cv2.putText(frame, "ENTRY logged", (30, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                last_seen_time = current_time
            else:
                if face_detected and (current_time - last_seen_time > EXIT_THRESHOLD_SEC):
                    face_detected = False
                    log_event("EXIT")
                    if HAS_DISPLAY:
                        cv2.putText(frame, "EXIT logged", (30, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # Reset the drowsiness counter on exit
                    counter_drowsy = 0
                    drowsy_alerted = False
                    eye_closed_since = None

            # ---- Skeletons (face/eye detection stays internal, no drawing) ----
            # Face + eyes are still detected for EAR/drowsiness/entry-exit,
            # but the face/eye skeleton drawing is removed.
            if pose_result and pose_result.pose_landmarks:
                draw_pose_skeleton(frame, pose_result.pose_landmarks[0], w, h)
            if object_result and object_result.detections:
                draw_objects(frame, object_result.detections)

            # ---- Part 2: Drowsiness detection (EAR) ----
            ear = None
            hud_ear = None; hud_dist = 0; hud_lbl = "Unknown"
            if face_present:
                landmarks = face_result.face_landmarks[0]
                left_ear = eye_aspect_ratio(landmarks, LEFT_EYE, w, h)
                right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
                ear = (left_ear + right_ear) / 2.0
                hud_ear = ear

                if ear < EAR_THRESHOLD:
                    counter_drowsy += 1
                    eyes_open_since = None  # eyes closed: cancel open-grace timer
                    # Track how long eyes have been closed (for the 3-second sound alarm)
                    if eye_closed_since is None:
                        eye_closed_since = current_time
                    closed_for = current_time - eye_closed_since

                    # Always show the live countdown
                    if HAS_DISPLAY:
                        cv2.putText(frame, f"Eyes closed: {closed_for:.1f}s / {DROWSINESS_ALARM_SEC:.1f}s", (30, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                    # Early visual alert (frame-based)
                    if counter_drowsy >= EAR_CONSEC_FRAMES and not drowsy_alerted:
                        drowsy_alerted = True
                        print_event("Drowsiness detected!")
                        if HAS_DISPLAY:
                            cv2.putText(frame, "DROWSINESS!", (30, 100),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                    # Sound alarm: eyes closed for 3 seconds -> play alarm sound
                    if ALARM_ENABLED and closed_for >= DROWSINESS_ALARM_SEC:
                        if current_time - last_alarm_time > ALARM_REPEAT_SEC:
                            last_alarm_time = current_time
                            print_event(f"WARNING: eyes closed for {closed_for:.1f}s - ALARM SOUND")
                            play_alarm_sound(freq=ALARM_BEEP_HZ, duration=5.0)
                        if HAS_DISPLAY:
                            cv2.putText(frame, "WARNING: DROWSINESS ALARM!", (30, 200),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                else:
                    # Eyes open: only reset the 3s timer after 0.6s of open eyes
                    if eyes_open_since is None:
                        eyes_open_since = current_time
                    if current_time - eyes_open_since > 0.6:
                        counter_drowsy = 0
                        drowsy_alerted = False
                        eye_closed_since = None
                        eyes_open_since = None

                # Show the live EAR value + estimated distance from camera
                dist_cm = estimate_distance_cm(landmarks, w)
                dist_lbl = distance_label(dist_cm)
                hud_dist = dist_cm; hud_lbl = dist_lbl
                if HAS_DISPLAY:
                    cv2.putText(frame, f"EAR: {ear:.2f}", (30, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    dist_col = (0, 255, 0) if dist_lbl in ("Good",) else (0, 255, 255) if dist_lbl in ("Close", "Far") else (0, 0, 255)
                    cv2.putText(frame, f"Distance: {dist_cm:.0f}cm ({dist_lbl})", (30, 180),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, dist_col, 2)
                    if dist_cm >= 120:
                        cv2.putText(frame, "TOO FAR!", (30, 220),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                elif frame_idx % 30 == 0:
                    # In text mode, print every 30 frames to reduce spam
                    print_event(f"Current EAR: {ear:.3f} | Distance: {dist_cm:.0f}cm ({dist_lbl})")

            # ---- Part 3: Hand skeletons (visual only, up to 2 hands) ----
            if hand_result.hand_landmarks:
                for hi, hand_landmarks in enumerate(hand_result.hand_landmarks[:2]):
                    raw = [(lm.x * w, lm.y * h) for lm in hand_landmarks]
                    hand_smooths[hi] = smooth_points(hand_smooths[hi], raw, alpha=0.65)
                    # Draw the advanced 21-point hand skeleton
                    draw_hand_skeleton(frame, hand_landmarks, w, h, smoothed_pts=hand_smooths[hi])
            else:
                hand_smooths = [None, None]

            # Sleek HUD bar (top) + hint (bottom) — part of the camera only
            if HAS_DISPLAY:
                entry_txt = "In" if face_detected else "Out"
                draw_hud(frame, hud_ear, hud_dist, hud_lbl, entry_txt, fps)

            # Show the window only when a display is available
            if HAS_DISPLAY:
                cv2.namedWindow("Smart Webcam - q to quit", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Smart Webcam - q to quit", 960, 720)
                cv2.imshow("Smart Webcam - q to quit", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    print_event("Stopped by user (pressed q)")
                    break
            else:
                # In headless mode, pause slightly to avoid hogging the CPU
                time.sleep(0.03)

    except KeyboardInterrupt:
        print_event("Stopped with Ctrl+C")
    finally:
        # Clean up resources
        cap.release()
        if HAS_DISPLAY:
            cv2.destroyAllWindows()
        for _obj in [face_landmarker, hand_landmarker, pose_landmarker, object_detector]:
            try:
                if _obj:
                    _obj.close()
            except Exception:
                pass
        print_event("Smart Webcam Assistant finished")


if __name__ == "__main__":
    main()
