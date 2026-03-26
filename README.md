# 🧠 AI-Powered Real-Time Surveillance System

This project is an **intelligent CCTV monitoring system** that combines **computer vision, deep learning, and vision-language models (VLMs)** to automatically analyze live video streams and detect suspicious activities in real time.

It integrates:

* Object detection (YOLOv8)
* Scene understanding (Vision Language Model via Ollama)
* Emotion recognition (DeepFace)
* Smart alert system (rule-based + AI reasoning)
* PyQt6 GUI for live monitoring

---

## 🚀 Features

### 🎯 Real-Time Object Detection

* Uses **YOLOv8** to detect objects like people, bags, and potential threats.
* Bounding boxes with confidence scores are drawn live on video.

### 🧠 Scene Description (VLM)

* Uses a Vision-Language Model (`moondream`) via Ollama.
* Generates human-like descriptions of what is happening in the frame.

Example:

```
"A man standing near a door holding an object"
```

---

### 😊 Emotion Detection

* Detects faces using OpenCV Haar Cascade.
* Uses DeepFace to identify emotions such as:

  * happy
  * angry
  * sad
  * fearful
* Helps enhance context understanding of situations.

---

### 🚨 Suspicious Activity Detection

Hybrid system:

1. **Rule-Based Detection**

   * Keywords like: `knife`, `weapon`, `gun`, etc.
2. **AI-Based Reasoning**

   * LLM evaluates scene description and returns:

     * YES → Suspicious
     * NO → Safe

---

### 🔔 Smart Alert System

* Displays:

  * 🚨 On-screen alert
  * Popup notification (PyQt)
* Includes **cooldown mechanism** to avoid alert spam

---

### 🖥️ Graphical User Interface

Built using **PyQt6**:

* Live video feed display
* Connect to:

  * Webcam
  * RTSP CCTV streams
* Activity history panel
* Clear history button

---

## 🏗️ Architecture

```
Camera Feed (Webcam / CCTV)
            │
            ▼
     Frame Capture
            │
 ┌──────────┼──────────┐
 ▼          ▼          ▼
YOLO     Emotion     VLM (Ollama)
Detect   Detection   Scene Description
            │          │
            └────┬─────┘
                 ▼
        Suspicious Check
       (Rules + AI Model)
                 │
        ┌────────┴────────┐
        ▼                 ▼
   Alert System     Description Log
```

---

## ⚙️ Tech Stack

* **Python**
* **OpenCV** – video processing
* **YOLOv8 (Ultralytics)** – object detection
* **DeepFace** – emotion recognition
* **Ollama (moondream model)** – scene understanding
* **PyQt6** – GUI interface
* **Threading** – non-blocking processing

---

## 🧩 Key Components

### `analyze_frame()`

* Captures frame
* Detects emotions
* Sends image to VLM
* Generates description
* Runs suspicious detection

---

### `check_suspicious()`

* Uses:

  * Keyword filtering
  * LLM reasoning
* Returns True/False

---

### `detect_emotions()`

* Detects faces
* Runs DeepFace analysis
* Returns dominant emotions

---

### GUI (`AIApp`)

Handles:

* Video streaming
* Buttons (Webcam / CCTV)
* Alert popups
* History tracking

---

## 🔌 Input Sources

* 📷 Webcam (default)
* 📡 RTSP CCTV stream

---

## 📊 Output

* Live annotated video feed
* Scene description overlay
* Emotion detection results
* Alert notifications
* Activity history log

---

## ⚠️ Limitations

* Emotion detection accuracy depends on face clarity
* VLM responses may vary
* Requires GPU for optimal performance
* Real-time performance depends on system specs

---

## 🔮 Future Improvements

* Face recognition (identity tracking)
* Multi-camera support
* Cloud alert system (SMS / Email)
* Weapon detection model fine-tuning
* Database logging

---

## 🧪 How to Run

```bash
pip install -r requirements.txt
python main.py
```

Make sure:

* Ollama is running
* Model `moondream` is installed
* YOLO weights are available

---

## 💡 Use Cases

* Smart CCTV monitoring
* Public safety systems
* Retail theft detection
* Office surveillance automation

---

## 👨‍💻 Author

Developed as an AI-powered surveillance prototype combining modern deep learning tools.

---

## ⭐ If you like this project

Give it a star ⭐ and feel free to contribute!
