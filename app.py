import time
from ultralytics import YOLO
import argostranslate.package as argos_pkg
import numpy as np
import os, json
from google import genai
from google.genai import types
import easyocr
import argostranslate.translate as argos
import torch
from PIL import Image
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import AutoModelForCausalLM
import io

app = Flask(__name__)
CORS(app)

MODEL_ID = "vikhyatk/moondream2"
REVISION = "2025-06-21"

print("Loading EchoVision AI...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    revision=REVISION,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map={"": "cuda"}
)

print("EchoVision AI loaded successfully!")
yolo = YOLO("yolov8s.pt")
print("YOLO loaded.")
@app.route("/")
def home():
    return jsonify({
        "message": "Welcome to EchoVision API",
        "status": "running"
    })
@app.route("/ask", methods=["POST"])
def ask():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image provided"}), 400

        question = request.form.get(
            "question", "Describe the surroundings."
        )

        image_file = request.files["image"]
        image = Image.open(image_file.stream).convert("RGB")

        result = model.query(image, question)

        return jsonify({
            "answer": result["answer"],
            "question": question
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/home")
def landing():
    return render_template("landing.html")

@app.route("/ui")
def ui():
    return render_template("index.html")
NAV_CLASSES = [0, 1, 2, 3, 5, 7, 9, 10, 11, 13, 16, 24, 28, 56, 57, 58, 59, 60]

@app.route("/detect", methods=["POST"])
def detect():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image provided"}), 400

        image = Image.open(request.files["image"].stream).convert("RGB")
        w, h = image.size
        result = yolo.predict(image, conf=0.5, classes=NAV_CLASSES, verbose=False)[0]

        objects = []
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx = (x1 + x2) / 2 / w
            area = ((x2 - x1) * (y2 - y1)) / (w * h)
            objects.append({
                "label": result.names[int(box.cls[0])],
                "direction": "on your left" if cx < 0.33 else "on your right" if cx > 0.66 else "ahead",
                             "distance": "very close" if (y2 - y1) / h > 0.65 else "close" if (y2 - y1) / h > 0.35 else "far",
                "area": round(area, 3)
            })

        objects.sort(key=lambda o: o["area"], reverse=True)
        return jsonify({"objects": objects})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
ocr_readers = {}
ARGOS_CODES = {"ch_sim": "zh", "ch_tra": "zh"}
def get_reader(lang):
    if lang not in ocr_readers:
        langs = ["en"] if lang == "en" else [lang, "en"]
        ocr_readers[lang] = easyocr.Reader(langs, gpu=True)
    return ocr_readers[lang]

def ensure_pack(src, tgt):
    langs = argos.get_installed_languages()
    s = next((l for l in langs if l.code == src), None)
    t = next((l for l in langs if l.code == tgt), None)
    if s and t and s.get_translation(t):
        return
    argos_pkg.update_package_index()
    p = next((x for x in argos_pkg.get_available_packages()
              if x.from_code == src and x.to_code == tgt), None)
    if p is None:
        raise ValueError(f"No Argos language pack for {src} -> {tgt}")
    argos_pkg.install_from_path(p.download())


CANDIDATES = ["ja", "hi", "ko", "ch_sim", "ar"]   # languages to detect between

def local_auto(arr):
    best_lang, best_texts, best_score = None, [], 0
    for lang in CANDIDATES:
        res = get_reader(lang).readtext(arr)
        # only non-English text counts toward the score
        score = sum(c for _, t, c in res if c >= 0.4 and t.strip() and not t.isascii())
        if score > best_score:
            best_lang, best_score = lang, score
            best_texts = [t for _, t, c in res if c >= 0.4 and t.strip()]
    return best_lang, best_texts


@app.route("/translate", methods=["POST"])
def translate():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image provided"}), 400

        lang = request.form.get("lang", "auto")
        target = request.form.get("target", "en")
        image = Image.open(request.files["image"].stream).convert("RGB")
        arr = np.array(image)

        if lang == "auto":
            lang, texts = local_auto(arr)
            if not lang:
                return jsonify({"lines": []})
        else:
            res = get_reader(lang).readtext(arr)
            texts = [t for _, t, c in res if c >= 0.4 and t.strip()]

        if not texts:
            return jsonify({"lines": []})

        src = ARGOS_CODES.get(lang, lang)
        if src == target:
            return jsonify({"lang": lang, "lines": [{"original": t, "translated": t} for t in texts]})

        if src != "en":
            ensure_pack(src, "en")
        if target != "en":
            ensure_pack("en", target)

        lines = []
        for t in texts:
            if t.isascii():
                lines.append({"original": t, "translated": t})
            else:
                lines.append({"original": t, "translated": argos.translate(t, src, target)})
        return jsonify({"lang": lang, "lines": lines})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
gclient = genai.Client() if os.environ.get("GEMINI_API_KEY") else None
GEMINI_MODELS = ["gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite"]

SIGN_PROMPT = (
    "Find all readable text in this image (signs, boards, menus, labels) in any language. "
    "Return ONLY JSON in this shape: "
    '{"signs":[{"language":"detected language","original":"text as written",'
    '"translation":"natural English translation",'
    '"explanation":"one or two plain sentences on what it means and what a visitor should do"}]}. '
    'If there is no text, return {"signs":[]}.'
)
GEMINI_MODELS = ["gemini-3.7-flash", "gemini-2.5-flash", "gemini-2.5-flash-lite"]

GEMINI_MODELS = ["gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite"]

def gemini_retry(contents, config, tries=2):
    errors = []
    for m in GEMINI_MODELS:
        for i in range(tries):
            try:
                return gclient.models.generate_content(
                    model=m, contents=contents, config=config)
            except Exception as e:
                errors.append(m + ": " + str(e)[:70])
                if "404" in str(e):
                    break
                time.sleep(2 ** i)
    raise RuntimeError(" | ".join(errors))
@app.route("/translate_ai", methods=["POST"])
def translate_ai():
    try:
        if gclient is None:
            return jsonify({"error": "GEMINI_API_KEY is not set"}), 500
        if "image" not in request.files:
            return jsonify({"error": "No image provided"}), 400

        f = request.files["image"]
        resp = gemini_retry(
    contents=[types.Part.from_bytes(data=f.read(), mime_type=f.mimetype or "image/jpeg"),
              SIGN_PROMPT],
    config=types.GenerateContentConfig(response_mime_type="application/json"),
)
        return jsonify(json.loads(resp.text))

    except Exception as e:
        return jsonify({"error": str(e)}), 500
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
