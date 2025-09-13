# GUI.py
import sys
import json
import cv2
import numpy as np
import time
import os
import csv
import face_recognition
from collections import defaultdict
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from ultralytics import YOLO
import logging
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger('ultralytics').setLevel(logging.WARNING)

# ---------------- Config ----------------
YOLO_MODEL = "yolov8n.pt"   # small & fast
KNOWN_FACES_DIR = "known_faces"  # folder with subfolders per person
LOG_FILE = "activity_log.csv"
CAMERA_KEYS_FILE = "urls.json"
FRAME_SKIP = 1  # process every FRAME_SKIP frames for face rec (helps speed)

# ---------------- Load YOLO ----------------
print("[INFO] Loading ...")
model = YOLO(YOLO_MODEL)

# ---------------- Load urls.json ----------------
if not os.path.exists(CAMERA_KEYS_FILE):
    print(f"[WARN] {CAMERA_KEYS_FILE} not found. Creating default webcam entry.")
    with open(CAMERA_KEYS_FILE, "w") as f:
        json.dump({"Webcam": [0]}, f)
with open(CAMERA_KEYS_FILE) as f:
    urls_dict = json.load(f)

# ---------------- Known faces ----------------
known_face_encodings = []
known_face_names = []

if os.path.isdir(KNOWN_FACES_DIR):
    for person_name in os.listdir(KNOWN_FACES_DIR):
        person_folder = os.path.join(KNOWN_FACES_DIR, person_name)
        if os.path.isdir(person_folder):
            for fn in os.listdir(person_folder):
                if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                    img_path = os.path.join(person_folder, fn)
                    try:
                        img = face_recognition.load_image_file(img_path)
                        enc = face_recognition.face_encodings(img)
                        if enc:
                            known_face_encodings.append(enc[0])
                            known_face_names.append(person_name)
                    except Exception as e:
                        print(f"[WARN] Failed to load {img_path}: {e}")

print(f"[INFO] Loaded {len(known_face_names)} known faces: {known_face_names}")

# ---------------- Activity tracking data structure ----------------
# Only tracking Vishal (or any known person) — extendable
activity_template = {
    "phone": {"active": False, "start": None, "total_time": 0.0, "count": 0},
    "laptop": {"active": False, "start": None, "total_time": 0.0, "count": 0},
    "chair": {"active": False, "start": None, "total_time": 0.0, "count": 0},
}
# will hold per-name data
people_activity = defaultdict(lambda: {k: v.copy() for k, v in activity_template.items()})

def update_activity_for_person(name, activity_key, detected):
    now = time.time()
    data = people_activity[name][activity_key]
    if detected:
        if not data["active"]:
            data["active"] = True
            data["start"] = now
            data["count"] += 1
    else:
        if data["active"]:
            data["active"] = False
            duration = now - (data["start"] or now)
            data["total_time"] += duration
            data["start"] = None

# ---------------- Logging on exit ----------------
def dump_logs_to_csv(camera_name="unknown"):
    exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["Name", "Date", "Phone Count", "Phone Time (s)", "Laptop Time (s)", "Chair Count", "Chair Time (s)"])
        for name, acts in people_activity.items():
            phone_total = round(acts["phone"]["total_time"], 2)
            laptop_total = round(acts["laptop"]["total_time"], 2)
            chair_total = round(acts["chair"]["total_time"], 2)
            writer.writerow([name, time.strftime("%Y-%m-%d"), acts["phone"]["count"], phone_total, laptop_total, acts["chair"]["count"], chair_total])
    print(f"[INFO] Activity logged to {LOG_FILE}")

# ---------------- Utility ----------------
def hash_to_color(label):
    h = hash(label) % (256*256*256)
    b = h % 256; g = (h//256)%256; r = (h//65536)%256
    return (int(b), int(g), int(r))

def bbox_iou(a, b):
    # a,b = (x1,y1,x2,y2)
    xa1, ya1, xa2, ya2 = a
    xb1, yb1, xb2, yb2 = b
    xi1 = max(xa1, xb1); yi1 = max(ya1, yb1)
    xi2 = min(xa2, xb2); yi2 = min(ya2, yb2)
    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0
    inter = (xi2 - xi1) * (yi2 - yi1)
    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

# ---------------- Worker & Signals ----------------
class WorkerSignals(QObject):
    frame = pyqtSignal(np.ndarray)
    stats = pyqtSignal(dict)

class VideoProcessingWorker(QRunnable):
    def __init__(self, urls):
        super().__init__()
        self.urls = urls
        self.signals = WorkerSignals()
        self.stop_flag = False
        self.frame_count = 0

    def run(self):
        # open captures once
        caps = []
        for url in self.urls:
            try:
                if isinstance(url, int) or (isinstance(url, str) and str(url).isdigit()):
                    caps.append(cv2.VideoCapture(int(url)))
                else:
                    caps.append(cv2.VideoCapture(url))
            except Exception as e:
                print(f"[ERROR] Opening {url}: {e}")

        # get default sizes from first cap
        width = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        height = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
        fps = caps[0].get(cv2.CAP_PROP_FPS) or 15
        delay = 1.0 / max(5, int(fps))

        while not self.stop_flag:
            frames = []
            for cap in caps:
                ret, frm = cap.read()
                if not ret:
                    frames.append(None)
                else:
                    frames.append(frm)
            # skip if none
            if all(f is None for f in frames):
                time.sleep(0.5)
                continue

            # For each non-None frame, process and build a grid (single camera common case)
            processed = []
            stats_emit = {}
            for frm in frames:
                if frm is None:
                    processed.append(np.zeros((height, width, 3), dtype=np.uint8))
                    continue

                # YOLO detection (single pass)
                res = model(frm, verbose=False)
                boxes = []
                names = []
                # res[0].boxes: x1,y1,x2,y2,conf,cls
                dets = res[0].boxes.data.cpu().numpy() if len(res) and hasattr(res[0].boxes, "data") else []
                for d in dets:
                    x1, y1, x2, y2, conf, cls = d
                    x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
                    label = model.names[int(cls)]
                    boxes.append((x1,y1,x2,y2))
                    names.append(label)

                # Face recognition (every FRAME_SKIP frames for speed)
                vishal_names = []  # names of known people seen in this frame
                face_locations = []
                face_names = []
                if (self.frame_count % FRAME_SKIP) == 0:
     # Step 1: Resize frame to 1/4 size for faster processing
                    small_frame = cv2.resize(frm, (0, 0), fx=0.25, fy=0.25)

    # Step 2: Convert from BGR (OpenCV) to RGB (face_recognition)
                    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Step 3: Detect face locations
                    flocs = face_recognition.face_locations(rgb_small_frame)

    # Step 4: Encode faces
                    fencs = face_recognition.face_encodings(rgb_small_frame, flocs)

    # Step 5: Compare and record names
                    for (t, r, b, l), enc in zip(flocs, fencs):
        # Scale back up to original frame size
                        top, right, bottom, left = [v * 4 for v in (t, r, b, l)]
                        face_locations.append((top, right, bottom, left))
                        matches = face_recognition.compare_faces(known_face_encodings, enc, tolerance=0.5)
                        name = "Unknown"
                        if True in matches:
                                first = matches.index(True)
                                name = known_face_names[first]
                                face_names.append(name)
                        if name != "Unknown":
                                vishal_names.append(name)

                # Now link recognized faces to person detection boxes (choose person bbox containing face center)
                person_indices = [i for i,lab in enumerate(names) if lab == "person"]
                person_boxes = [boxes[i] for i in person_indices]
                person_map = {}  # name -> person_box index
                for floc, fname in zip(face_locations, face_names):
                    if fname == "Unknown":
                        continue
                    ftop, fright, fbot, fleft = floc
                    fcx = (fleft + fright)//2; fcy = (ftop + fbot)//2
                    for idx, pb in zip(person_indices, person_boxes):
                        x1,y1,x2,y2 = pb
                        if x1 <= fcx <= x2 and y1 <= fcy <= y2:
                            person_map[fname] = pb
                            break

                # Determine which objects overlap the Vishal's person box
                detected_for_vishal = {"phone": False, "laptop": False, "chair": False}
                # map YOLO class labels to our keys (some models use 'cell phone' word)
                for box, label in zip(boxes, names):
                    lab = label.lower()
                    # define object types by substring
                    if "phone" in lab or "cell phone" in lab or "mobile" in lab:
                        obj_key = "phone"
                    elif "laptop" in lab or "keyboard" in lab or "monitor" in lab:
                        obj_key = "laptop"
                    elif "chair" in lab:
                        obj_key = "chair"
                    else:
                        obj_key = None

                    if obj_key:
                        # if Vishal mapped to a person box, check overlap, otherwise count if any vishal name present
                        if person_map:
                            # pick first recognized person (e.g., Vishal)
                            for pname, pbox in person_map.items():
                                # ensure it's the person we want (we track all known names)
                                iou = bbox_iou(box, pbox)
                                if iou > 0.05:  # small overlap threshold
                                    detected_for_vishal[obj_key] = True
                        else:
                            # no person mapping but known face seen somewhere — if that known name was present, we allow detection
                            if vishal_names:
                                detected_for_vishal[obj_key] = True

                # Update activity for every recognized known person in this frame
                for seen_name in set(vishal_names):
                    for act in detected_for_vishal:
                        update_activity_for_person(seen_name, act, detected_for_vishal[act])

                # Draw boxes & labels on frame
                draw_frame = frm.copy()
                # draw object boxes
                for (x1,y1,x2,y2), lbl in zip(boxes, names):
                    color = hash_to_color(lbl)
                    cv2.rectangle(draw_frame, (x1,y1), (x2,y2), color, 2)
                    cv2.putText(draw_frame, lbl, (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                # draw face boxes and names
                for (top,right,bottom,left), fname in zip(face_locations, face_names):
                    cv2.rectangle(draw_frame, (left, top), (right, bottom), (0,255,0), 2)
                    cv2.putText(draw_frame, fname, (left, top-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

                # prepare stats to emit
                stats_emit = {}
                for name in set(known_face_names):
                    acts = people_activity[name]
                    stats_emit[name] = {
                        "phone_count": acts["phone"]["count"],
                        "phone_time": round(acts["phone"]["total_time"] + (time.time()-acts["phone"]["start"] if acts["phone"]["active"] and acts["phone"]["start"] else 0), 1),
                        "laptop_time": round(acts["laptop"]["total_time"] + (time.time()-acts["laptop"]["start"] if acts["laptop"]["active"] and acts["laptop"]["start"] else 0), 1),
                        "chair_count": acts["chair"]["count"],
                        "chair_time": round(acts["chair"]["total_time"] + (time.time()-acts["chair"]["start"] if acts["chair"]["active"] and acts["chair"]["start"] else 0), 1)
                    }

                processed.append(draw_frame)
                self.frame_count += 1

                       # build output frame
            if len(processed) == 1:
                out_frame = processed[0]
            else:
                # choose grid size depending on view mode
                if hasattr(self, "view_mode") and self.view_mode in ["2x2", "4x4", "8x8"]:
                    grid_map = {"2x2": 2, "4x4": 4, "8x8": 8}
                    grid_size = grid_map[self.view_mode]
                else:
                    grid_size = int(np.ceil(np.sqrt(len(processed))))

                ph = height * grid_size
                pw = width * grid_size
                out_frame = np.zeros((ph, pw, 3), dtype=np.uint8)
                for i, f in enumerate(processed):
                    r = i // grid_size
                    c = i % grid_size
                    if r < grid_size and c < grid_size:
                        out_frame[r*height:(r+1)*height, c*width:(c+1)*width] = f


            # emit frame and stats
            try:
                self.signals.frame.emit(out_frame)
                self.signals.stats.emit(stats_emit)
            except Exception:
                pass

            time.sleep(delay)

        for cap in caps:
            cap.release()

    def stop(self):
        self.stop_flag = True

# ---------------- PyQt GUI ----------------
class VideoWidget(QWidget):
    def __init__(self, parent=None):
        super(VideoWidget, self).__init__(parent)
        self.image = None

    def set_image(self, image):
        self.image = image
        self.update()

    def paintEvent(self, event):
        if self.image is not None:
            painter = QPainter(self)
            painter.drawImage(self.rect(), self.image)

class DetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CCTV Activity Dashboard")
        self.setGeometry(50, 50, 1400, 800)

        central = QWidget()
        self.setCentralWidget(central)
        h = QHBoxLayout(central)

        # --- Video Display ---
        self.video_widget = VideoWidget()
        self.video_widget.setMinimumSize(960, 540)
        h.addWidget(self.video_widget, 3)

        # --- Right Panel ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setAlignment(Qt.AlignTop)
        h.addWidget(right_panel, 1)

        # --- Camera Switch UI ---
        cam_label = QLabel("<b>Select Camera:</b>")
        right_layout.addWidget(cam_label)

        self.cam_selector = QComboBox()
        self.cam_selector.addItems(list(urls_dict.keys()))
        right_layout.addWidget(self.cam_selector)

        self.switch_btn = QPushButton("Switch Camera")
        right_layout.addWidget(self.switch_btn)
        self.switch_btn.clicked.connect(self.switch_camera)

      

        # --- Stats ---
        stats_label = QLabel("<b>Activity Stats:</b>")
        right_layout.addWidget(stats_label)

        self.stats_labels = {}
        for name in known_face_names:
            title = QLabel(f"<b>{name}</b>")
            phone_lbl = QLabel("Phone: 0 uses / 0s")
            laptop_lbl = QLabel("Laptop: 0s")
            chair_lbl = QLabel("Chair: 0 uses / 0s")
            right_layout.addWidget(title)
            right_layout.addWidget(phone_lbl)
            right_layout.addWidget(laptop_lbl)
            right_layout.addWidget(chair_lbl)
            right_layout.addSpacing(10)
            self.stats_labels[name] = (phone_lbl, laptop_lbl, chair_lbl)

        # --- Control Buttons ---
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        right_layout.addLayout(btn_layout)

        self.start_btn.clicked.connect(self.start_stream)
        self.stop_btn.clicked.connect(self.stop_stream)

        # worker/thread
        self.worker = None
        self.threadpool = QThreadPool()
        self.current_camera = None
                # --- View Mode Buttons ---
        view_label = QLabel("<b>View Mode:</b>")
        right_layout.addWidget(view_label)

        self.view_buttons = {}
        grid_layout = QHBoxLayout()
        for mode in ["Single", "2x2", "4x4", "8x8"]:
            btn = QPushButton(mode)
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, m=mode: self.set_view_mode(m))
            grid_layout.addWidget(btn)
            self.view_buttons[mode] = btn
        right_layout.addLayout(grid_layout)

        self.view_mode = "Single"

    def start_stream(self):
        if self.view_mode == "Single":
            selected_cam = self.cam_selector.currentText()
            urls = urls_dict[selected_cam]
            self.current_camera = selected_cam
        else:
            # flatten all urls for grid mode
            urls = []
            for name, ulist in urls_dict.items():
                urls.extend(ulist)
            self.current_camera = "All Cameras"

        self.worker = VideoProcessingWorker(urls)
        self.worker.signals.frame.connect(self.update_frame)
        self.worker.signals.stats.connect(self.update_stats)
        self.threadpool.start(self.worker)

    def switch_camera(self):
        # stop current worker
        if self.worker:
            self.worker.stop()
            dump_logs_to_csv()
            self.worker = None

        # start new one
        self.start_stream()
    def set_view_mode(self, mode):
        # uncheck other buttons
            for m, btn in self.view_buttons.items():
             btn.setChecked(m == mode)
            self.view_mode = mode

        # restart worker if already running
            if self.worker:
                 self.worker.stop()
                 dump_logs_to_csv()
                 self.worker = None
                 self.start_stream()
    def stop_stream(self):
        if self.worker:
            self.worker.stop()
            dump_logs_to_csv()
            self.worker = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def update_frame(self, cv_img):
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_widget.set_image(qt_img)

    def update_stats(self, stats):
        for name, (phone_lbl, laptop_lbl, chair_lbl) in self.stats_labels.items():
            if name in stats:
                s = stats[name]
                phone_lbl.setText(f"Phone: {s['phone_count']} uses / {s['phone_time']}s")
                laptop_lbl.setText(f"Laptop: {s['laptop_time']}s")
                chair_lbl.setText(f"Chair: {s['chair_count']} uses / {s['chair_time']}s")

    def closeEvent(self, event):
        if self.worker:
            self.worker.stop()
            dump_logs_to_csv()
        event.accept()


# ---------------- Run ----------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = DetectionApp()
    win.show()
    sys.exit(app.exec_())
