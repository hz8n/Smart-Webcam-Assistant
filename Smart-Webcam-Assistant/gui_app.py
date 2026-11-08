"""
Smart Webcam Assistant - GUI App (CAMERA ONLY)
================================================
Full app, but the window shows ONLY the camera (no controls, no panels).
Features running internally:
  - Entry/Exit logging (webcam_log.csv)
  - Drowsiness detection + sound alarm (5s) after eyes closed 3s
  - Skeletons (visual only): hand (21, advanced) • pose (33, no face) • objects (cup...)
  - Face/eye detection internal only (for EAR), no face/eye drawing

Run:
  source venv/bin/activate
  python gui_app.py
Close: press Q or Esc, or the X button.
"""
import csv
import datetime
import os
import subprocess
import threading
import time
import urllib.request
import wave

import cv2

# Requirements: opencv-python, mediapipe, numpy, Pillow

# ---------------- Default settings ----------------
EAR_THRESHOLD = 0.25
EAR_CONSEC_FRAMES = 20
DROWSINESS_ALARM_SEC = 3.0   # sound alarm after eyes closed this long
EXIT_THRESHOLD_SEC = 5
LOG_FILE = "webcam_log.csv"
CAMERA_INDEX = 0
FACE_MODEL = "face_landmarker.task"
HAND_MODEL = "hand_landmarker.task"
POSE_MODEL = "pose_landmarker.task"
OBJECT_MODEL = "object_detector.tflite"
FACE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
OBJECT_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite"
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (17, 19), (15, 21),
    (12, 14), (14, 16), (16, 18), (18, 20), (14, 22),
    (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (27, 29),
    (24, 26), (26, 28), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
]


def timestamp_now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_models():
    models = [
        (FACE_MODEL, FACE_MODEL_URL), (HAND_MODEL, HAND_MODEL_URL),
        (POSE_MODEL, POSE_MODEL_URL), (OBJECT_MODEL, OBJECT_MODEL_URL),
    ]
    for path, url in models:
        if not os.path.exists(path):
            try:
                urllib.request.urlretrieve(url, path)
            except Exception:
                return False
    return True


def eye_aspect_ratio(landmarks, eye_indices, img_w, img_h):
    import numpy as np
    pts = []
    for idx in eye_indices:
        lm = landmarks[idx]
        pts.append((int(lm.x * img_w), int(lm.y * img_h)))
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
    import numpy as np
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
    """Draw a sleek HUD bar (top) + hint bar (bottom) — no extra widgets."""
    h, w, _ = frame.shape
    # Top bar: translucent black
    bar = frame.copy()
    cv2.rectangle(bar, (0, 0), (w, 46), (0, 0, 0), -1)
    cv2.addWeighted(bar, 0.55, frame, 0.45, 0, frame)
    # Accent line
    cv2.line(frame, (0, 46), (w, 46), (0, 255, 0), 2)
    ear_s = f"{ear:.2f}" if ear is not None else "--"
    dist_s = f"{dist:.0f}cm" if dist and dist > 0 else "--"
    line1 = f"EAR {ear_s}   Dist {dist_s} ({dist_lbl})   {entry_text}   {fps:.0f} FPS"
    # color by distance
    dist_col = (0, 255, 0) if dist_lbl == "Good" else (0, 255, 255) if dist_lbl in ("Close", "Far") else (255, 255, 255)
    cv2.putText(frame, line1, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, dist_col, 2)
    # Bottom hint bar
    bar2 = frame.copy()
    cv2.rectangle(bar2, (0, h - 36), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(bar2, 0.45, frame, 0.55, 0, frame)
    cv2.line(frame, (0, h - 36), (w, h - 36), (255, 255, 255), 1)
    cv2.putText(frame, "Q / Esc close  -  eyes closed 3s = 5s alarm", (12, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


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


def draw_objects(frame, detections, img_w, img_h):
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


def log_event(event_type):
    now = timestamp_now()
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["type", "timestamp"])
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([event_type, now])
    return now


def _beep_sync(freq=1200, duration=5.0):
    """Play a loud LONG alarm tone (blocking). Never raises."""
    try:
        import numpy as _np
        sr = 44100
        # One long continuous loud tone for `duration` seconds (default 5s)
        t = _np.linspace(0, duration, int(sr * duration), endpoint=False)
        samples = (_np.sin(2 * _np.pi * freq * t) * 0.9 * 32767).astype(_np.int16)
        wav_path = f"/tmp/drowsy_alarm_{os.getuid()}.wav"
        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
            wf.writeframes(samples.tobytes())
        for cmd in (["paplay", wav_path], ["aplay", wav_path],
                    ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", wav_path],
                    ["mpv", "--no-video", wav_path]):
            try:
                r = subprocess.run(cmd, timeout=8,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if r.returncode == 0:
                    return True
            except Exception:
                continue
    except Exception:
        pass
    try:
        r = subprocess.run(["beep", "-f", str(freq), "-l", str(int(duration * 1000))],
                           timeout=6, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if r.returncode == 0:
            return True
    except Exception:
        pass
    try:
        import sys
        print("\a", file=sys.stderr, end="", flush=True)
        return True
    except Exception:
        return False


def play_alarm_sound(freq=1200, duration=5.0):
    """Play the drowsiness alarm in a background thread (non-blocking)."""
    threading.Thread(target=lambda: _beep_sync(freq=freq, duration=duration), daemon=True).start()


class CameraOnlyApp:
    """Window that shows ONLY the camera. No panels, no buttons, no sliders."""

    def __init__(self, root):
        import tkinter as tk
        self.root = root
        root.title("Camera")
        # Camera fills the window: large but slightly smaller than full screen
        root.geometry("960x720")
        root.minsize(640, 480)
        root.maxsize(1100, 820)
        # No margins: video fills the whole window
        self.video_lbl = tk.Label(root, bg="black")
        self.video_lbl.pack(fill="both", expand=True)
        root.rowconfigure(0, weight=1)
        root.columnconfigure(0, weight=1)
        # Close with Q or Esc
        root.bind("<Key-q>", lambda e: self.on_close())
        root.bind("<Key-Q>", lambda e: self.on_close())
        root.bind("<Escape>", lambda e: self.on_close())
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        root.focus_set()

        # Internal state (no UI for these)
        self.running = True
        self.cap = None
        self.face_landmarker = None
        self.hand_landmarker = None
        self.pose_landmarker = None
        self.object_detector = None
        self.face_detected = False
        self.last_seen = time.time()
        self.counter_drowsy = 0
        self.drowsy_alerted = False
        self.eye_closed_since = None
        self.eyes_open_since = None
        self.last_alarm_time = 0
        self.hand_smooths = [None, None]
        self.frame_idx = 0
        self.fps = 0.0
        self._last_t = time.time()
        self._cached_pose = None
        self._cached_obj = None

        # Start camera automatically
        self.thread = threading.Thread(target=self.camera_loop, daemon=True)
        self.thread.start()

    def on_close(self):
        self.running = False
        try:
            self.root.destroy()
        except Exception:
            pass

    # ---------- Camera loop (separate thread) ----------
    def camera_loop(self):
        import numpy as np
        import mediapipe as mp
        from PIL import Image, ImageTk
        from mediapipe.tasks.python.core.base_options import BaseOptions
        from mediapipe.tasks.python.vision import (
            FaceLandmarker, FaceLandmarkerOptions,
            HandLandmarker, HandLandmarkerOptions,
            PoseLandmarker, PoseLandmarkerOptions,
            ObjectDetector, ObjectDetectorOptions,
        )
        from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

        if not ensure_models():
            return

        face_opts = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=FACE_MODEL),
            running_mode=VisionTaskRunningMode.IMAGE,
            num_faces=1, min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5, min_tracking_confidence=0.5,
        )
        hand_opts = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=HAND_MODEL),
            running_mode=VisionTaskRunningMode.IMAGE,
            num_hands=2, min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7, min_tracking_confidence=0.7,
        )
        pose_opts = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=POSE_MODEL),
            running_mode=VisionTaskRunningMode.IMAGE,
            num_poses=1, min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5, min_tracking_confidence=0.5,
        )
        obj_opts = ObjectDetectorOptions(
            base_options=BaseOptions(model_asset_path=OBJECT_MODEL),
            running_mode=VisionTaskRunningMode.IMAGE,
            score_threshold=0.4, max_results=5,
        )
        self.face_landmarker = FaceLandmarker.create_from_options(face_opts)
        self.hand_landmarker = HandLandmarker.create_from_options(hand_opts)
        try:
            self.pose_landmarker = PoseLandmarker.create_from_options(pose_opts)
        except Exception:
            self.pose_landmarker = None
        try:
            self.object_detector = ObjectDetector.create_from_options(obj_opts)
        except Exception:
            self.object_detector = None

        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self.cap.isOpened():
            return

        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = np.ascontiguousarray(rgb)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # Efficient: face+hand every frame, pose/object every 3rd frame
            face_res = self.face_landmarker.detect(mp_img)
            hand_res = self.hand_landmarker.detect(mp_img)
            self.frame_idx += 1
            if self.frame_idx % 3 == 0:
                if self.pose_landmarker:
                    self._cached_pose = self.pose_landmarker.detect(mp_img)
                if self.object_detector:
                    self._cached_obj = self.object_detector.detect(mp_img)
            pose_res = self._cached_pose
            obj_res = self._cached_obj

            now_t = time.time()
            # FPS (EMA)
            if self._last_t:
                dt = max(0.001, now_t - self._last_t)
                self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt) if self.fps else (1.0 / dt)
            self._last_t = now_t
            face_present = bool(face_res.face_landmarks)
            hud_ear = None; hud_dist = 0; hud_lbl = "Unknown"

            # --- Entry/Exit logging (internal) ---
            if face_present:
                if not self.face_detected:
                    self.face_detected = True
                    log_event("ENTRY")
                    cv2.putText(frame, "ENTRY", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                self.last_seen = now_t
            else:
                if self.face_detected and (now_t - self.last_seen > EXIT_THRESHOLD_SEC):
                    self.face_detected = False
                    log_event("EXIT")
                    cv2.putText(frame, "EXIT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    self.counter_drowsy = 0; self.drowsy_alerted = False; self.eye_closed_since = None
            if self.face_detected:
                cv2.putText(frame, "ENTRY", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # --- Pose skeleton (body only, no face) ---
            if pose_res and pose_res.pose_landmarks:
                draw_pose_skeleton(frame, pose_res.pose_landmarks[0], w, h)

            # --- Object detection: precise boxes (cup, etc.) ---
            if obj_res and obj_res.detections:
                draw_objects(frame, obj_res.detections, w, h)

            # --- Drowsiness + 3s -> 5s sound alarm (internal, always on) ---
            if face_present:
                lms = face_res.face_landmarks[0]
                ear = (eye_aspect_ratio(lms, LEFT_EYE, w, h) + eye_aspect_ratio(lms, RIGHT_EYE, w, h)) / 2.0
                dist_cm = estimate_distance_cm(lms, w)
                dist_lbl = distance_label(dist_cm)
                hud_ear = ear; hud_dist = dist_cm; hud_lbl = dist_lbl
                cv2.putText(frame, f"EAR: {ear:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                dist_col = (0, 255, 0) if dist_lbl == "Good" else (0, 255, 255) if dist_lbl in ("Close", "Far") else (0, 0, 255)
                cv2.putText(frame, f"Distance: {dist_cm:.0f}cm ({dist_lbl})", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, dist_col, 2)
                if dist_cm >= 120:
                    cv2.putText(frame, "TOO FAR!", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                if ear < EAR_THRESHOLD:
                    self.counter_drowsy += 1
                    self.eyes_open_since = None
                    if self.eye_closed_since is None:
                        self.eye_closed_since = now_t
                    closed_for = now_t - self.eye_closed_since
                    cv2.putText(frame, f"Eyes closed: {closed_for:.1f}s / {DROWSINESS_ALARM_SEC:.1f}s", (10, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    if self.counter_drowsy >= EAR_CONSEC_FRAMES and not self.drowsy_alerted:
                        self.drowsy_alerted = True
                    if closed_for >= DROWSINESS_ALARM_SEC:
                        if now_t - self.last_alarm_time > 5.5:
                            self.last_alarm_time = now_t
                            play_alarm_sound(freq=1200, duration=5.0)
                        cv2.putText(frame, "WARNING: DROWSINESS ALARM!", (10, 180),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                else:
                    if self.eyes_open_since is None:
                        self.eyes_open_since = now_t
                    if now_t - self.eyes_open_since > 0.6:
                        self.counter_drowsy = 0; self.drowsy_alerted = False; self.eye_closed_since = None
                        self.eyes_open_since = None

            # --- Hand skeletons (visual only, up to 2 hands) ---
            if hand_res.hand_landmarks:
                for hi, hl in enumerate(hand_res.hand_landmarks[:2]):
                    raw = [(lm.x * w, lm.y * h) for lm in hl]
                    self.hand_smooths[hi] = smooth_points(self.hand_smooths[hi], raw, alpha=0.65)
                    # Always draw the advanced skeleton when a hand is visible
                    draw_hand_skeleton(frame, hl, w, h, smoothed_pts=self.hand_smooths[hi])
            else:
                self.hand_smooths = [None, None]

            # Sleek HUD bar (top) + hint (bottom) — part of the camera only
            entry_txt = "In" if self.face_detected else "Out"
            draw_hud(frame, hud_ear, hud_dist, hud_lbl, entry_txt, self.fps)

            # Show only the camera, filling the window completely (no margins)
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            ww, hh = self.video_lbl.winfo_width(), self.video_lbl.winfo_height()
            if ww > 1 and hh > 1:
                img = img.resize((ww, hh))  # stretch to fill the window fully
            else:
                sw = self.root.winfo_screenwidth() or 960
                sh = self.root.winfo_screenheight() or 720
                img = img.resize((sw, sh))
            photo = ImageTk.PhotoImage(image=img)
            self.video_lbl.config(image=photo, text="")
            self.video_lbl.image = photo

        if self.cap:
            self.cap.release()
        for obj in [self.face_landmarker, self.hand_landmarker, self.pose_landmarker, self.object_detector]:
            try:
                if obj:
                    obj.close()
            except Exception:
                pass


if __name__ == "__main__":
    if not os.environ.get("DISPLAY"):
        print(f"[{timestamp_now()}] Error: no graphical display (DISPLAY is empty) - the GUI needs a display.")
        raise SystemExit(1)
    import tkinter as tk
    r = tk.Tk()
    CameraOnlyApp(r)
    r.mainloop()
