import os
import base64
import sqlite3
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
from datetime import datetime
from tensorflow.keras.models import load_model
from urllib.request import urlretrieve

app = Flask(__name__)
model = load_model('super_emotion_brain.h5')

EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Download face detector if missing
cascade_path = 'haarcascade_frontalface_default.xml'
if not os.path.exists(cascade_path):
    urlretrieve('https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml', cascade_path)
face_cascade = cv2.CascadeClassifier(cascade_path)

# DB
DB_PATH = 'emotions.db'
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''CREATE TABLE IF NOT EXISTS records
                    (name TEXT, image_path TEXT, emotion TEXT, confidence REAL, mode TEXT, timestamp TEXT)''')
    conn.commit()
    conn.close()
init_db()

UPLOADS = 'uploads'
os.makedirs(UPLOADS, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    name = request.form['name'].strip()
    mode = request.form['mode']
    img_b64 = request.form['image'].split(',')[1]
    img_bytes = base64.b64decode(img_b64)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    img_path = os.path.join(UPLOADS, f"{ts}.jpg")
    with open(img_path, 'wb') as f:
        f.write(img_bytes)

    gray = cv2.imread(img_path, 0)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0:
        return jsonify({'error': 'No face detected! Try better lighting.'})
    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
    face = cv2.resize(gray[y:y+h, x:x+w], (48, 48)).astype('float32') / 255
    pred = model.predict(np.expand_dims(np.expand_dims(face, 0), -1), verbose=0)
    emotion_idx = np.argmax(pred)
    emotion = EMOTIONS[emotion_idx]
    conf = f"{np.max(pred) * 100:.1f}"
    
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT INTO records VALUES (?, ?, ?, ?, ?, ?)", 
                 (name, img_path, emotion, float(conf), mode, datetime.now().isoformat()))
    conn.commit()
    conn.close()
    return jsonify({'emotion': emotion, 'confidence': conf, 'image': img_path})

@app.route('/history_data')
def history_data():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT * FROM records ORDER BY timestamp DESC LIMIT 50").fetchall()
    conn.close()
    return jsonify(rows)

@app.route('/uploads/<path:filename>')
def get_upload(filename):
    return send_from_directory(UPLOADS, filename)

if __name__ == '__main__':
    app.run(debug=True)