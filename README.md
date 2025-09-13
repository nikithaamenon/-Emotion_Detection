# 🎭 Real-Time Emotion Detection using OpenCV & MediaPipe

A lightweight, real-time emotion detection system using **OpenCV** and **MediaPipe**, designed to classify facial expressions like 😊 Happy, 😢 Sad, 😠 Angry, 😲 Surprised, and without using any machine learning or deep learning models. It works by analyzing the geometry of facial landmarks from a webcam feed using simple rules.

---

## 📌 Features

- 🔍 Facial landmark detection with MediaPipe  
- 🧠 Rule-based emotion classification  
- ⚙️ No training, datasets, or ML models required  
- 💻 Works live from your webcam  
- 🪶 Lightweight and beginner-friendly  

---

## 🛠️ Technologies Used

- Python  
- OpenCV  
- MediaPipe  

---

## 🧠 How It Works

1. MediaPipe detects **468 facial landmarks** in real-time from the webcam.  
2. Specific points on the **mouth, eyes, and eyebrows** are monitored.  
3. A set of **rule-based conditions** determine the emotion:  
   - 😊 **Happy** → Wide mouth + curved lips + open eyes  
   - 😢 **Sad** → Downturned lips + relaxed eyes  
   - 😠 **Angry** → Tight lips + furrowed brows  
   - 😲 **Surprised** → Raised brows + wide eyes  
   

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/emotion-detection-mediapipe.git
cd emotion-detection-mediapipe

INSTALL REQUIREMENTS:
pip install opencv-python mediapipe

🌱 Future Enhancements
Add more nuanced emotions like fear, disgust, or confusion

Replace rule-based logic with a lightweight ML classifier (optional)

Add emoji overlays, voice response, or animated feedback

Build a browser-based version using Flask or Streamlit

Enable face normalization for better accuracy
