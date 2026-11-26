# EchoVision

A voice-driven AI assistant that helps visually impaired people understand their surroundings, and helps travellers read foreign-language signs. It uses an ordinary webcam and runs its vision models locally on a consumer GPU.

## Features
- **Hands-free voice control:** say "start", "stop", "guide me" or "translate this sign".
- **Normal mode:** ask anything about the scene. Moondream 2 answers in short sentences using left, right and ahead.
- **Navigation mode:** YOLOv8 scans continuously and speaks short warnings such as "Person ahead, very close". It stays silent when nothing changes.
- **Sign translation:** reads signs, detects the language and speaks the English meaning. Works offline with EasyOCR and Argos Translate, with optional Gemini for explanations.
- **Interruptible speech:** the mic stays on while it talks, so you can cut in or say "stop".
- **Natural narration:** answers are rewritten to sound human, with start and stop beeps.
- **Live chat log** on the dashboard for sighted helpers.

## How it works
1. The browser captures a webcam frame (max 640 px JPEG) and listens through the Web Speech API.
2. The frame goes to the Flask backend: `/ask` (Moondream 2), `/detect` (YOLOv8), `/translate` (EasyOCR + Argos) or `/translate_ai` (Gemini).
3. The answer is humanized, shown in chat and spoken aloud.

## Tech stack
Python 3.11, PyTorch (CUDA), Transformers, Moondream 2, Ultralytics YOLOv8, EasyOCR, Argos Translate, Google Gemini (optional), Flask, Pillow, HTML/CSS/JavaScript with Web Speech and Web Audio APIs.

## Project structure
```
EchoVision/
├── app.py
├── requirements.txt
├── templates/      landing page and dashboard
├── scripts/        test scripts
├── docs/           diagrams and screenshots
└── samples/        local test images (not tracked)
```

## Setup
```
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
set GEMINI_API_KEY=your_key    (optional)
python app.py
```
Open http://127.0.0.1:5000/home in Chrome or Edge, click Start, and say "start".

Requires an NVIDIA GPU (developed on an RTX 4050, 6 GB).

## Limitations
- Moondream 2 is small and can be wrong about positions, counts and distance.
- Distance in Navigation mode is estimated from box size, not measured.
- Chrome and Edge speech recognition may send audio to a cloud service.
- Offline sign translation covers a limited set of languages.
- This is an awareness aid. It does not replace a cane, a guide dog or the user's own judgement.

## Author
Ayushman Panadiya, B.Tech AI and ML, Galgotias College of Engineering and Technology.