# SignConnect – AI Sign Language to Text Translator

SignConnect is a complete, real-time AI Sign Language translation web application powered by the pre-trained **Sign Transformer** (`sign_transformer.h5`) model. It captures live webcam video, extracts 3D hand landmarks via MediaPipe, normalizes features over a 30-frame sequence buffer, and performs inference through a high-performance FastAPI backend.

---

## 🚀 Features

- **Live Webcam Recognition**: 30 FPS video feed with on-canvas skeletal hand-tracking visualization.
- **Pre-Trained Sign Transformer**: Uses the authentic `sign_transformer.h5` model with 21 sign classes.
- **MediaPipe Landmark Normalization**: Computes 126 spatial features per frame (wrist-centered & L2-norm scaled for up to 2 hands) across a 30-frame sequence buffer.
- **Real-Time Confidence Scoring**: Displays top predicted sign, confidence gauge, and top-3 candidate distribution.
- **Natural Language Processing & AI Sentence Prediction**:
  - **⚡ Computational NLP Engine**: Offline, real-time linguistic Gloss-to-Text (G2T) engine that synthesizes subject-verb copulas, polite formulations, and temporal clauses without requiring any API keys.
  - **🧠 Generative AI Model (Groq LLaMA 3.1)**: Full generative AI sentence generation that translates sign language glosses into fluid spoken sentences and supports multilingual translation (English, Spanish, French, German, Hindi, Tamil).
  - **🔤 Raw Sign Stream**: Direct raw token stream for verbatim gloss inspection.
- **Controls & Accessibility**: Full suite of controls: Predict Sentence, Add Space, Backspace, Clear, Copy, and Native Text-to-Speech (TTS).
- **Error Handling**: Graceful handling for camera permissions, missing hands, and offline backend status.

---

## 📂 Project Structure

```
SignConnect/
├── backend/
│   ├── main.py                      # FastAPI application entrypoint & static mount
│   ├── requirements.txt             # Python backend dependencies
│   ├── .env.example                 # Environment configuration template
│   ├── models/
│   │   ├── sign_transformer.h5      # PRIMARY pre-trained Keras model
│   │   └── sign_transformer.tflite  # TFLite alternative / fallback
│   └── services/
│       ├── model_service.py         # Model loading, validation, and inference
│       └── sentence_service.py      # Local rule-based sentence construction
├── frontend/
│   ├── index.html                   # Modern responsive UI
│   ├── styles.css                   # Glassmorphism dark styling & layout
│   └── app.js                       # MediaPipe hand tracker & UI controllers
└── README.md                        # Documentation & setup guide
```

---

## 🔤 Supported Sign Vocabulary (21 Classes)

`AFTERNOON`, `ALRIGHT`, `DRESS`, `GOOD`, `HE`, `HELLO`, `I`, `IT`, `MONSOON`, `MORNING`, `PLEASE`, `RICH`, `SAD`, `SHE`, `SUMMER`, `T-SHIRT`, `TELEPHONE`, `THANK YOU`, `TODAY`, `WINTER`, `YOU`

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.10+ (tested on Python 3.12)
- Modern web browser (Chrome, Edge, Firefox) with webcam support

### Step 1: Navigate to Backend Directory
```powershell
cd "C:\Users\Thrishiga Sree\.gemini\antigravity\scratch\SignConnect\backend"
```

### Step 2: Install Dependencies
```powershell
pip install -r requirements.txt
```

> **Note on Model Compatibility**: The model `sign_transformer.h5` was originally saved with Keras 2 (`2.12.0`). Modern TensorFlow 2.16+ / 2.21 defaults to Keras 3. The backend uses `tf-keras` to guarantee 100% backward compatibility and fidelity without altering the original model weights.

---

## ▶️ Running the Application

### Start the FastAPI Server:
```powershell
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Access the Web App:
Open your browser and navigate to:
```
http://localhost:8000
```

- Interactive Web Interface: `http://localhost:8000`
- Swagger API Docs: `http://localhost:8000/docs`
- Health Endpoint: `http://localhost:8000/health`

---

## 🧪 Testing and Verification

To run backend inference unit tests:
```powershell
python -c "
import requests, numpy as np
dummy_seq = np.random.randn(30, 126).tolist()
res = requests.post('http://localhost:8000/predict', json={'input': dummy_seq})
print('Status:', res.status_code, 'Result:', res.json())
"
```
