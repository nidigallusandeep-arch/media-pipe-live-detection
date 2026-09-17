# 🤖 MediaPipe Computer Vision

A real-time Computer Vision application built using **Python, MediaPipe and Streamlit**.

This project provides live **Face Detection, Hand Detection, Pose Detection, and combined detection** through a browser camera using WebRTC.

---

## 🚀 Live Demo

🔗 **Streamlit App:**  
Add your deployed Streamlit app URL here.

---

## 📌 Project Overview

The goal of this project is to build a real-time computer vision application that can detect and track:

- 👤 Face landmarks
- ✋ Hand landmarks
- 🧍 Human pose landmarks
- 🤖 Face + Hand + Pose simultaneously

The application uses the user's webcam and processes video frames in real time.

---

## ✨ Features

### 👤 Face Detection
- Detects facial landmarks
- Displays facial landmark points
- Supports multiple faces

### ✋ Hand Detection
- Detects hand landmarks
- Detects up to 2 hands
- Draws hand connections and landmark points

### 🧍 Pose Detection
- Detects human body landmarks
- Tracks major body joints
- Displays pose skeleton

### 🤖 All Detection
Runs:

- Face Detection
- Hand Detection
- Pose Detection

at the same time.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming Language |
| MediaPipe | Computer Vision |
| OpenCV | Image Processing |
| NumPy | Numerical Operations |
| Streamlit | Web Application |
| Streamlit-WebRTC | Live Webcam Streaming |
| PyAV | Video Frame Processing |
| Git & GitHub | Version Control |

---

## 📂 Project Structure

```text
media-pipe-live-detection/
│
├── app.py
├── face_detection.py
├── hand_detection_1.py
├── pose_detection.py
│
├── face_landmarker.task
├── hand_landmarker.task
├── pose_landmarker_full.task
│
├── requirements.txt
├── test_face.py
├── .gitignore
└── README.md
