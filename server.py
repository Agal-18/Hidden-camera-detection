from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import base64
import os
import gdown

# Create Flask app
app = Flask(__name__)

# Load trained model
model_path = "spycam_cnn_final.h5"

if not os.path.exists(model_path):
    url = "https://drive.google.com/uc?id=1ijGOH5gTxnRLSmBbcUx1RsDh_uEjleBZ"
    gdown.download(url, model_path, quiet=False)

model = tf.keras.models.load_model(model_path, compile=False)


# ---------- PAGES ----------

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


# Detection page after login/signup
@app.route("/detect_page")
def detect_page():
    return render_template("index.html")


# ---------- DETECTION API ----------

@app.route("/detect", methods=["POST"])
def detect():

    try:
        data = request.json["image"]

        img_data = data.split(",")[1]

        img_bytes = base64.b64decode(img_data)

        npimg = np.frombuffer(img_bytes, np.uint8)

        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        # IMPORTANT → model input size
        frame = cv2.resize(frame,(224,224), interpolation=cv2.INTER_AREA)

        frame = frame.astype("float32") / 255.0

        frame = np.expand_dims(frame, axis=0)

        prediction = model(frame, training=False)

        score = float(prediction[0][0])

        if score > 0.5:
            result = "⚠️ SPY CAMERA DETECTED"
        else:
            result = "✅ NO SPY CAMERA DETECTED"

        return jsonify({"result": result})

    except Exception as e:

        print("Detection Error:", e)

        return jsonify({"result": "Detection Error"})


# ---------- RUN SERVER ----------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)