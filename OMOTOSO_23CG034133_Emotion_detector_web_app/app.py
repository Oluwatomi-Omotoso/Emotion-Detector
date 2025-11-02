import streamlit as st
from PIL import Image
import numpy as np
import os
import io
import sqlite3
from datetime import datetime

# ML imports
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing.image import img_to_array

try:
    from deepface import DeepFace

    deepface_available = True
except Exception:
    deepface_available = False

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "emotion_cnn.h5")
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
DB_DIR = os.path.join(BASE_DIR, "database")
DB_PATH = os.path.join(DB_DIR, "app_usage.db")

# Ensure Paths
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)

# Page setup
st.set_page_config(page_title="Emotion Detection App", layout="centered")
st.title("🎭 Real-time Emotion Detection")
st.markdown("Upload a photo or use your webcam to detect emotions. (Name is optional)")

# Try to load model
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = load_model(MODEL_PATH)
        st.info("Loaded local trained model.")
    except Exception:
        st.error("Found model file but failed to load.")
        model = None
else:
    st.info(
        "No local model found at models/emotion_cnn.h5. Falling back to DeepFace if available."
    )


# DB functions
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            timestamp TEXT,
            image_path TEXT,
            predicted_emotion TEXT,
            scores TEXT
        )
    """
    )
    conn.commit()
    conn.close()


def log_usage(name, image_path, predicted_emotion, scores_dict):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """
        INSERT INTO usage (name, timestamp, image_path, predicted_emotion, scores)
        VALUES (?, ?, ?, ?, ?)
    """,
        (
            name,
            datetime.utcnow().isoformat(),
            image_path,
            predicted_emotion,
            str(scores_dict),
        ),
    )
    conn.commit()
    conn.close()


init_db()

# UI input
name = st.text_input("Your name (optional)")
option = st.radio("Choose input source:", ["📸 Webcam", "📁 Upload an Image"])
if option == "📸 Webcam":
    img_file_buffer = st.camera_input("Take a photo")
else:
    img_file_buffer = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])


def predict_with_local_model(pil_img, model):
    img_resized = pil_img.convert("L").resize((48, 48))
    arr = img_to_array(img_resized) / 255.0
    arr = np.expand_dims(arr, axis=0)  # (1,48,48,1)
    if arr.shape[-1] != 1:
        arr = np.expand_dims(arr, -1)
    preds = model.predict(arr)[0]
    CLASS_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
    scores = {label: float(preds[i] * 100) for i, label in enumerate(CLASS_LABELS)}
    dominant = CLASS_LABELS[int(np.argmax(preds))]
    return dominant, scores


if img_file_buffer is not None:
    try:
        img = Image.open(img_file_buffer).convert("RGB")
    except Exception:
        img = Image.open(io.BytesIO(img_file_buffer.read())).convert("RGB")
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    filename = f"{name if name else 'anon'}_{timestamp}.jpg".replace(" ", "_")
    save_path = os.path.join(UPLOAD_DIR, filename)
    img.save(save_path)

    predicted_emotion = None
    confidence_scores = {}

    if model is not None:
        try:
            predicted_emotion, confidence_scores = predict_with_local_model(img, model)
        except Exception:
            st.error("Local model prediction failed.")

    if (predicted_emotion is None or not confidence_scores) and deepface_available:
        try:
            arr = np.array(img)
            result = DeepFace.analyze(arr, actions=["emotion"], enforce_detection=False)
            res = result[0] if isinstance(result, list) else result
            predicted_emotion = res.get("dominant_emotion", "unknown")
            confidence_scores = res.get("emotion", {})
        except Exception:
            st.error("DeepFace prediction failed.")

    if predicted_emotion is None:
        st.error(
            "No model available to make a prediction. Please place a Keras model at models/emotion_cnn.h5 or ensure DeepFace is installed."
        )
    else:
        st.image(
            img,
            caption=f"Predicted Emotion: {predicted_emotion}",
            use_container_width=True,
        )
        st.markdown(f"### 🧠 Detected Emotion: **{predicted_emotion}**")
        with st.expander("See confidence scores"):
            for emotion, score in confidence_scores.items():
                try:
                    score_val = float(score)
                except:
                    score_val = score
                if score_val <= 1.0:
                    score_val *= 100.0
                st.write(f"{emotion}: {score_val:.2f}%")
        try:
            log_usage(name, save_path, predicted_emotion, confidence_scores)
            st.success("Usage logged to local database.")
        except Exception:
            st.error("Failed to log usage.")

# Show basic stats / recent entries
if st.button("Show recent usage (last 10)"):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT id, name, timestamp, image_path, predicted_emotion FROM usage ORDER BY id DESC LIMIT 10"
    )
    rows = c.fetchall()
    conn.close()
    if rows:
        for r in rows:
            st.write(f"{r[0]} | {r[1]} | {r[2]} | {r[4]}")
    else:
        st.write("No usage logged yet.")
