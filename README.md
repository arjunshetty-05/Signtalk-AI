# SignTalk AI — Complete Developer & Onboarding Guide

> **Project Name:** SignTalk AI (Bidirectional AI Communication Bridge for Hearing & Speech Impaired)  
> **Institution:** RNS Institute of Technology, Bengaluru — Dept. of Information Science & Engineering  
> **Academic Year:** 2025–2026 | **Batch:** #24  
> **Target:** Final-Year B.E. Capstone Project, Hackathon/Demo Ready Accessibility Platform & IEEE Publication  

---

## 1. Executive Summary & Core Objective

SignTalk AI bridges the two-way communication gap between hearing/speech-impaired individuals and non-signing individuals:
1. **Deaf/Mute to Hearing**: Real-time sign language gestures captured via webcam $\rightarrow$ 17-point MoveNet pose extraction $\rightarrow$ 30-frame BiLSTM sequence classification $\rightarrow$ Facial emotion detection (DeepFace) $\rightarrow$ LLM/NLP sentence reconstruction & grammar correction (Gemini 2.0 Flash / Flan-T5) $\rightarrow$ Multilingual translation (English, Hindi, Kannada) $\rightarrow$ Audio synthesis (TTS) & visual subtitle rendering.
2. **Hearing to Deaf/Mute**: Spoken audio stream $\rightarrow$ Real-time speech-to-text (Whisper Small) $\rightarrow$ Immediate live visual captions & conversation log.
3. **Cloud & Offline Resilience**: Firebase Firestore syncs cross-client dialogues and analytics; a unified `X-Offline-Mode` fallback ensures zero-network operation with local BiLSTM TFLite models, Flan-T5, offline phrasebooks, and Coqui TTS.

---

## 2. System Architecture & High-Level Dataflow

```
   [ WEBCAM INPUT ]                                [ MICROPHONE INPUT ]
          │ (20-30 fps frames)                              │ (WebM/Opus audio chunks)
          ▼                                                 ▼
  [ MoveNet Thunder ] (17 Keypoints)               [ Whisper Small STT ] (/ws/speech)
          │                                                 │
  [ Normalization & Smoothing ]                             ▼
          │                                        [ Live Hearing Subtitles ]
          ▼                                                 │
  [ 30-Frame Rolling Buffer ]                               ▼
          │                                        [ Conversation History ]
          ▼
  [ BiLSTM Classifier ] (runs/exp_top40_stratified)
          │ (Top gesture label, debounced)
          ├────────────────────────┐
          ▼                        ▼
  [ Facial Emotion ]       [ Raw Gesture Tokens ]
     (DeepFace 7-cls)              │
          │                        │
          └───────────┬────────────┘
                      ▼
            [ NLP Correction ]
     (Primary: Gemini 2.0 Flash | Offline: Flan-T5)
                      │
                      ▼
           [ Grammatical Sentence ]
                      │
          ┌───────────┴────────────┐
          ▼                        ▼
  [ Translation Engine ]    [ Text-to-Speech ]
   (Google Translate /       (gTTS / Coqui TTS)
    Offline Phrasebook)            │
          │                        ▼
          ▼                 [ Speaker Output ]
  [ React Subtitle Bar ]
```

---

## 3. Technology Stack & Key Dependencies

| Subsystem | Primary Technology | Fallback / Offline / Alternate |
|---|---|---|
| **Pose / Keypoint Extraction** | TensorFlow Hub MoveNet Thunder (`singlepose/thunder/4`, 17 keypoints) | Cached local TFLite (`models/movenet.tflite`) |
| **Gesture Sequence Model** | TensorFlow / Keras Bidirectional LSTM (128 units, Dropout 0.3) | Quantized Float16 TFLite (`models/bilstm.tflite`) |
| **Facial Emotion Recognition** | DeepFace (7 classes: happy, sad, angry, fear, surprise, neutral, disgust) | Evaluated every 10th frame; defaults to `neutral` |
| **NLP Sentence Correction** | Google Gemini 2.0 Flash (`google-generativeai`) | Local Hugging Face `google/flan-t5-small` |
| **Speech-to-Text (STT)** | OpenAI Whisper Small (local `openai-whisper` / `transformers`) | Web Audio API / browser native STT |
| **Text-to-Speech (TTS)** | Google Text-to-Speech (`gTTS`) | Coqui TTS (`your_tts` offline model) |
| **Multilingual Translation** | Google Cloud Translation API (`en`, `hi`, `kn`) | Local in-memory LRU $\rightarrow$ Firestore $\rightarrow$ `phrasebook.json` |
| **Backend & Realtime** | FastAPI, Uvicorn, Python-SocketIO, WebSockets | Docker, Docker Compose |
| **Auth & Database** | Firebase Authentication (JWT/Bearer), Firestore, Firebase Storage | Mock dev token verification |
| **Web Frontend** | React.js (Vite), Tailwind CSS, Framer Motion, Axios, Recharts, Lucide | Responsive glassmorphic UI |
| **Mobile App (Optional)** | Flutter, Riverpod, TFLite Flutter | Cross-platform mobile (Android/iOS) |

---

## 4. Locked Interface Contracts (DO NOT BREAK)

1. **Keypoint Extraction Count**: Exactly **17 keypoints** (MoveNet Thunder standard), **never 33** (MediaPipe standard).
2. **Gesture Classifier Signature**:
   ```python
   def classify_sequence(sequence: np.ndarray) -> dict[str, Any]:
       # Input shape: (30, 17, 2)
       # Output: {"label": str, "confidence": float}
   ```
3. **Stabilization Rules**:
   - 3+ consecutive matching predictions required before emitting.
   - 1.5-second cooldown before the same label can emit again.
4. **WebSocket `/ws/gesture` Protocol**:
   - Client sends: `{ "frame": "base64_encoded_jpeg" }` at 20–30 FPS.
   - Server returns: `{"label": str, "confidence": float, "timestamp": float}` and `{"type": "corrected_sentence", "sentence": str, "source": str, "low_confidence": bool}`.
5. **Shared Offline Mode**:
   - Triggered via HTTP header `X-Offline-Mode: true` or query param `?offline=true`.
   - Handled uniformly in `backend/api/core/offline_mode.py`.

---

## 5. Critical Gotchas & Immediate Fixes

1. **Unmirrored Webcam Orientation**:
   - In `frontend/src/components/WebcamView.jsx`, react-webcam must **NOT** have `mirrored={true}` enabled. Flipping pixels changes left/right hand coordinates and drops recognition accuracy.
2. **WebSocket Frame Skip (`frame_skip=2`)**:
   - Real sign gestures take ~2.5–3.0 seconds. At 20 FPS, 30 frames is only 1.5 seconds. The WebSocket connection URL must pass `?frame_skip=2` so the 30-frame buffer spans 3 seconds.
3. **MoveNet Cache Corruption**:
   - If backend crashes at startup with `ValueError: ... contains neither saved_model.pb nor saved_model.pbtxt`, clear `%TEMP%\tfhub_modules`.
4. **Translation API Key & GCP Billing**:
   - If `GOOGLE_TRANSLATE_API_KEY` is not set or billing is inactive, translation falls back to original text or `phrasebook.json`. Ensure offline fallback doesn't throw 500s.
5. **Firebase Service Account**:
   - Development requires `backend/secrets/firebase-service-account.json`. If missing, verify `backend/api/firebase/firebase_client.py` handles mock mode cleanly.

---

## 6. Local Quickstart (How to Run Everything)

### Prerequisites
- Python 3.10+ (Virtual environment in `backend/venv`)
- Node.js 18+ & npm
- Git

### Backend Setup
```bash
cd backend
# Windows:
.\venv\Scripts\activate
# Install deps (if fresh):
pip install -r requirements.txt
# Copy environment file:
cp .env.example .env

# Run FastAPI with Socket.IO:
uvicorn api.socket_manager:socket_app --host 127.0.0.1 --port 8000 --reload
```
Swagger Documentation: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### Frontend Setup
```bash
cd frontend
npm install
npm run dev
```
Web Application: [http://localhost:3000](http://localhost:3000)

---

## 7. Model Training & Datasets

- **Preprocessed Dataset**: Located in `backend/data/sequences/` and mapped via `backend/data/labels.csv` (INCLUDE dataset, 4,257 sequences).
- **Curated 40-Word Model**:
  ```bash
  python train_bilstm.py --data_dir data/sequences --labels_csv data/labels_top40.csv --split_mode stratified --output_dir runs/exp_top40_stratified
  ```
- **TFLite Conversion**:
  ```bash
  python convert_to_tflite.py --saved_model runs/exp_top40_stratified/saved_model --output models/bilstm.tflite
  ```

---

## 8. Prioritized Task List (For the Next Few Days)

1. [ ] **Verify Live Gesture Recognition**: Test with `frame_skip=2` and unmirrored camera; confirm latency < 250ms.
2. [ ] **Fix Firebase Credentials / Mock Mode**: Ensure smooth login & conversation history retrieval even without live GCP billing.
3. [ ] **Gemini Prompt Tuning**: Refine prompt in `backend/nlp_correction.py` for snappy, natural conversational outputs.
4. [ ] **Translation & Indic Languages**: Verify English $\rightarrow$ Hindi / Kannada translation and speech playback.
5. [ ] **Finalize Presentation / Paper Assets**: Export confusion matrices from `runs/`, record UI walkthrough, and compile IEEE Phase-2 report.
