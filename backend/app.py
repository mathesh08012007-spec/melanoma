from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
import io

app = Flask(__name__)
CORS(app)

model = None

# ── Load Model ─────────────────────────────────────────
import os
from flask import send_file

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "..", "vit_model.keras")

def load_model():
    global model
    try:
        from vit_keras import vit
        from tensorflow.keras.models import load_model as km

        model = km(MODEL_PATH, compile=False, safe_mode=False)
        print("✅ ViT model loaded.")

    except Exception as e:
        print(f"⚠️ Model load failed: {e}")

@app.route("/")
def index():
    html_path = os.path.join(BASE_DIR, "..", "frontend", "melanoma_app.html")
    return send_file(html_path)

# ── Prediction API ─────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        from PIL import Image
        img = Image.open(io.BytesIO(file.read())).convert("RGB").resize((224, 224))
        arr = np.expand_dims(np.array(img) / 255.0, axis=0)
    except Exception as e:
        return jsonify({"error": f"Image processing failed: {e}"}), 500

    prob = None

    try:
        if model is not None:
            raw = model.predict(arr)
            if raw.shape[-1] == 1:
                prob = float(raw[0][0])
            else:
                prob = float(raw[0][1])
        else:
            import random
            prob = random.uniform(0, 1)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500

    if prob is None:
        return jsonify({"error": "Prediction failed"}), 500

    return jsonify({
        "label": "Melanoma ⚠️" if prob > 0.5 else "Benign ✅",
        "confidence": round(prob * 100, 2),
        "is_melanoma": prob > 0.5,
        "demo_mode": model is None
    })

# ── Run Server ─────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)