from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import base64
import os
import gdown

app = Flask(__name__)

# ---------------- MODEL LOADING ----------------

model = None
model_path = "spycam_cnn_final.h5"

def load_cnn_model():
    global model

    # download model if not present
    if not os.path.exists(model_path):
        url = "https://drive.google.com/uc?id=1ijGOH5gTxnRLSmBbcUx1RsDh_uEjleBZ"
        gdown.download(url, model_path, quiet=False)

    # load model safely
    try:
        from keras.models import load_model
        model = load_model(model_path, compile=False)
        print("Model loaded successfully")

    except Exception as e:
        print("Model loading failed:", e)

# load model at startup
load_cnn_model()

# ---------------- PAGES ----------------

@app.route("/")
def home():
    return render_template("login.html")

@app.route("/login")
def login():
    return render_template("login.html")

@app.route("/signup")
def signup():
    return render_template("signup.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/detect_page")
def detect_page():
    return render_template("index.html")

# ---------------- DETECTION ----------------

@app.route("/detect", methods=["POST"])
def detect():

    try:
        data = request.json["image"]

        img_data = data.split(",")[1]
        img_bytes = base64.b64decode(img_data)

        npimg = np.frombuffer(img_bytes, np.uint8)

        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        frame = cv2.resize(frame, (224,224))

        frame = frame.astype("float32") / 255.0
        frame = np.expand_dims(frame, axis=0)

        prediction = model.predict(frame)

        score = float(prediction[0][0])

        if score > 0.5:
            result = "⚠️ SPY CAMERA DETECTED"
        else:
            result = "✅ NO SPY CAMERA DETECTED"

        return jsonify({"result": result})

    except Exception as e:
        print("Detection Error:", e)
        return jsonify({"result": "Detection Error"})

# ---------------- RUN SERVER ----------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)