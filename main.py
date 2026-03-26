import sys
import cv2
import threading
import ollama
import tempfile
import os
import textwrap
import time
from ultralytics import YOLO
from deepface import DeepFace
from datetime import datetime
from PyQt6.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QTextEdit, QPushButton, QHBoxLayout, QMessageBox
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QTimer

# ==============================
# CONFIG
# ==============================
MODEL_NAME = "moondream:latest"
ANALYSIS_INTERVAL = 1

latest_description = "Starting..."
processing = False
last_time = 0
description_history = []
HISTORY_LIMIT = 20

# 🔔 Alert control
last_alert_time = 0
ALERT_COOLDOWN = 5  # seconds

# ==============================
# LOAD YOLO
# ==============================
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

yolo = YOLO("yolov8n.pt")
yolo.to(device)

print(f"YOLO running on: {device}")

# ==============================
# SUSPICIOUS DETECTION
# ==============================
def check_suspicious(text):
    try:
        danger_keywords = ["knife", "blade", "weapon", "holding a knife", "armed", "holding a gun", "holding a weapon"]

        text_lower = text.lower()
        if any(word in text_lower for word in danger_keywords):
            print("🔴 KNIFE DETECTED (RULE-BASED)")
            return True

        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{
                'role': 'user',
                'content': f"""
You are a CCTV security AI.

Analyze the following scene description and determine if it is suspicious or dangerous.

Description:
{text}

Rules:
- Respond ONLY with one word: YES or NO
- YES = violence, threat, crime, weapon, panic, danger, knife
- NO = normal safe activity
"""
            }],
            options={"num_predict": 2}
        )

        answer = response['message']['content'].strip().upper()
        print("🔍 Suspicious Analysis:", answer)

        return "YES" in answer

    except Exception as e:
        print("Suspicious check error:", e)
        return False

# ==============================
# SAVE FRAME
# ==============================
def save_temp_image(frame):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    cv2.imwrite(temp_file.name, frame)
    return temp_file.name

# ==============================
# EMOTION DETECTION
# ==============================
def detect_emotions(frame):
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(30, 30)
        )

        if len(faces) == 0:
            return "No face detected"

        emotions = []

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face = cv2.resize(face, (160, 160))

            analysis = DeepFace.analyze(
                face,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='opencv'
            )

            if isinstance(analysis, list):
                emotions.extend([a['dominant_emotion'] for a in analysis])
            else:
                emotions.append(analysis['dominant_emotion'])

        return ", ".join(list(set(emotions)))

    except Exception as e:
        print("Emotion error:", e)
        return "No emotion"

# ==============================
# ANALYZE FRAME
# ==============================
def analyze_frame(frame):
    global latest_description, processing, description_history, last_time

    if processing:
        return

    try:
        processing = True

        emotions_text = detect_emotions(frame)

        image_path = save_temp_image(frame)

        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{
                'role': 'user',
                'content': 'Describe what is happening?',
                'images': [image_path]
            }],
            options={"num_predict": 20}
        )

        vlm_text = response['message']['content']
        timestamp = datetime.now().strftime("%H:%M:%S")

        latest_description = f"{timestamp} | {vlm_text} | Emotion: {emotions_text}"

        description_history.append(latest_description)
        if len(description_history) > HISTORY_LIMIT:
            description_history.pop(0)

        if check_suspicious(latest_description):
            print("🚨 ALERT TRIGGERED!")
            print("Detected Text:", latest_description)

        os.remove(image_path)

    except Exception as e:
        print("Analyze error:", e)
        latest_description = f"Error: {str(e)}"

    finally:
        processing = False

# ==============================
# DRAW FUNCTIONS
# ==============================
def draw_yolo_boxes(frame):
    try:
        results = yolo(frame)
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                label = yolo.names[cls]
                conf = float(box.conf[0])

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}",
                            (x1, max(20,y1-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0,255,0), 2)
    except Exception as e:
        print("YOLO error:", e)

    return frame

def draw_alert(frame, text):
    if check_suspicious(text):
        cv2.putText(frame, "🚨 ALERT!", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0,0,255), 3)
    return frame

def draw_description_box(frame, text):
    h, w, _ = frame.shape

    wrapped = textwrap.wrap(text, width=70)

    line_height = 25
    padding = 20
    box_height = line_height * len(wrapped) + padding
    box_height = min(box_height, int(h * 0.4))

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - box_height), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    max_lines = int((box_height - padding) / line_height)
    wrapped = wrapped[-max_lines:]

    y = h - box_height + 30
    for line in wrapped:
        cv2.putText(frame, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0,255,0), 2)
        y += line_height

    return frame

def draw_datetime(frame):
    t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, t, (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255,255,255), 2)
    return frame

# ==============================
# GUI
# ==============================
class AIApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Surveillance")
        self.resize(1200, 800)

        layout = QHBoxLayout()
        self.setLayout(layout)

        self.video_label = QLabel()
        self.video_label.setFixedSize(960,720)
        layout.addWidget(self.video_label)

        side = QVBoxLayout()

        self.rtsp_input = QTextEdit()
        self.rtsp_input.setPlaceholderText("Enter RTSP URL (CCTV)...")
        self.rtsp_input.setFixedHeight(40)
        side.addWidget(self.rtsp_input)

        self.connect_btn = QPushButton("Connect CCTV")
        self.connect_btn.clicked.connect(self.connect_cctv)
        side.addWidget(self.connect_btn)

        self.webcam_btn = QPushButton("Use Webcam")
        self.webcam_btn.clicked.connect(self.use_webcam)
        side.addWidget(self.webcam_btn)

        self.history = QTextEdit()
        self.history.setReadOnly(True)
        side.addWidget(self.history)

        btn = QPushButton("Clear")
        btn.clicked.connect(self.clear_history)
        side.addWidget(btn)

        layout.addLayout(side)

        self.cap = cv2.VideoCapture(0)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    # 🔔 POPUP FUNCTION
    def show_alert_popup(self, text):
        msg = QMessageBox(self)
        msg.setWindowTitle("🚨 Security Alert")
        msg.setText("Suspicious activity detected!")
        msg.setInformativeText(text)
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg.exec()

    def connect_cctv(self):
        url = self.rtsp_input.toPlainText().strip()
        if not url:
            print("❌ No RTSP URL")
            return

        new_cap = cv2.VideoCapture(url)

        if not new_cap.isOpened():
            print("❌ Failed to connect CCTV")
            return

        self.cap.release()
        self.cap = new_cap
        print("✅ CCTV Connected")

    def use_webcam(self):
        self.cap.release()
        self.cap = cv2.VideoCapture(0)

    def clear_history(self):
        global description_history
        description_history = []
        self.history.clear()

    def update_frame(self):
        global last_time, last_alert_time

        ret, frame = self.cap.read()
        if not ret:
            return

        frame_small = cv2.resize(frame, (224,224))

        now = time.time()
        if now - last_time >= ANALYSIS_INTERVAL and not processing:
            last_time = now
            threading.Thread(
                target=analyze_frame,
                args=(frame_small.copy(),)
            ).start()

        display = cv2.resize(frame, (960,720))
        display = draw_yolo_boxes(display)
        display = draw_description_box(display, latest_description)
        display = draw_alert(display, latest_description)
        display = draw_datetime(display)

        # 🔔 ALERT POPUP WITH COOLDOWN
        if check_suspicious(latest_description):
            now = time.time()
            if now - last_alert_time > ALERT_COOLDOWN:
                last_alert_time = now
                QTimer.singleShot(0, lambda: self.show_alert_popup(latest_description))

        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        h,w,ch = rgb.shape
        qt_img = QImage(rgb.data, w,h,ch*w, QImage.Format.Format_RGB888)

        self.video_label.setPixmap(QPixmap.fromImage(qt_img))

        self.history.setText("\n".join(description_history[-HISTORY_LIMIT:]))

    def closeEvent(self, event):
        self.cap.release()
        event.accept()

# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = AIApp()
    win.show()
    sys.exit(app.exec())