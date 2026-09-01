🤟 VoiceSignAI

Multilingual, real-time sign-language communication that turns hand
gestures into understandable speech.

VoiceSignAI is a hackathon-focused assistive communication platform that
combines MediaPipe hand tracking, landmark-based sign
recognition, temporal stability filtering, AI sentence
refinement, text-to-speech, personalized custom signs, and a
multilingual dataset/API layer.

The goal is simple:

Sign → Recognize → Stabilize → Build Sentence → Improve → Speak

🌟 Why VoiceSignAI?

Communication barriers can make everyday interactions difficult for
people who use sign language, especially when the other person does not
understand it.

VoiceSignAI provides an accessible bridge by:

👋 Detecting hand landmarks from a camera

🧠 Recognizing static and dynamic gesture patterns

🧹 Filtering unstable/duplicate predictions before they become words

📝 Building a sentence from recognized signs

✨ Optionally polishing the sentence with AI while preserving its
meaning

🔊 Converting the resulting sentence into speech

✍️ Allowing users to teach the system their own custom signs

🌍 Supporting ASL, ISL, BSL, and personalized custom vocabulary

🆘 Providing a persistent HELP/SOS detection layer

🧪 Including demo/mock paths for presentations and environments
without a trained model

🚀 Core Experience

1. Live Sign → Voice

The main interface provides:

Language selection

Live camera or snapshot input

Hand landmark extraction

Gesture classification

Confidence + temporal stability filtering

Sentence accumulation

AI sentence improvement

Voice output

The application also provides quick gesture triggers and a full demo
sequence, which are useful during a hackathon presentation.

2. ✍️ Custom Sign Studio --- "Teach My Sign"

Users can create personalized vocabulary by recording a sign multiple
times.

The system:

Requires at least 3 samples

Recommends 5 samples

Supports static poses and dynamic sequences

Converts recordings into canonical landmark features

Normalizes/resamples dynamic sequences

Stores custom signs and samples in SQLite

Lets users list, update, add samples to, and delete custom signs

Uses stored custom samples during recognition/fallback matching

This makes the system adaptable rather than limited to a fixed
vocabulary.

3. 📖 Multilingual Vocabulary

The backend supports:

🇺🇸 ASL --- American Sign Language

🇮🇳 ISL --- Indian Sign Language

🇬🇧 BSL --- British Sign Language

✍️ CUSTOM --- Personalized signs

The included sample datasets contain both static and dynamic landmark
samples for ASL, ISL, and BSL.

4. 🆘 Help / SOS Detection

A dedicated recognition monitor watches for labels such as:

help, emergency, assistance, need help, danger, and sos

The safety detector requires the signal to persist for 5 seconds
before triggering a HELP_DETECTED recognition event, reducing
accidental alerts from a single noisy prediction.

🧠 AI & Recognition Architecture

                   ┌─────────────────────┐
                   │   Camera / Image    │
                   └──────────┬──────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │ MediaPipe Hands     │
                   │ 21 landmarks/hand   │
                   │ up to 2 hands       │
                   └──────────┬──────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │ Canonical 126-D     │
                   │ Landmark Vector     │
                   └──────────┬──────────┘
                              │
                ┌─────────────┴─────────────┐
                │                           │
                ▼                           ▼
       Static Pose Path             Dynamic Sequence Path
       126 features                 30 × 126 features
                │                           │
                └─────────────┬─────────────┘
                              ▼
                   ┌─────────────────────┐
                   │ Recognition Engine  │
                   │ ML / heuristic /    │
                   │ custom fallback     │
                   └──────────┬──────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │ Confidence + Motion │
                   │ + Debouncing        │
                   └──────────┬──────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │ Sentence Builder    │
                   │ Raw gesture words   │
                   └──────────┬──────────┘
                              │
                    Explicit user action
                              │
                              ▼
                   ┌─────────────────────┐
                   │ AI Sentence Polish  │
                   │ OpenAI / Gemini     │
                   └──────────┬──────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │ Text-to-Speech      │
                   │ gTTS / browser TTS  │
                   └──────────┬──────────┘
                              │
                              ▼
                        🔊 Spoken Voice

🏗️ Project Architecture

The repository contains two complementary application layers:

Streamlit application --- the hackathon-facing interactive UI in
app.py

FastAPI service layer --- structured recognition, dataset,
custom-sign, Gemini, and WebSocket APIs under src/api/

High-level structure

VoiceSignAI/
│
├── app.py                         # Main Streamlit application
├── run_server.py                  # FastAPI + Uvicorn launcher
├── config.py                      # Sign2Voice pipeline configuration
├── requirements.txt
├── .env.example
├── hand_landmarker.task           # MediaPipe hand landmark model
│
├── ai/
│   └── sentence_corrector.py      # OpenAI sentence refinement
│
├── backend/
│   ├── gesture_processor.py       # Confidence, stability, cooldown, dedup
│   ├── sentence_builder.py        # Raw + AI sentence state
│   └── state_manager.py           # Thread-safe application state
│
├── speech/
│   └── tts.py                     # gTTS MP3 generation
│
├── integration/
│   ├── pipeline.py                # Main Sign2Voice orchestration layer
│   └── ml_adapter.py              # Normalizes ML output formats
│
├── src/
│   ├── api/                       # FastAPI routes + WebSocket
│   ├── custom_signs/              # Teach My Sign management
│   ├── ingestion/                 # ASL/ISL/BSL dataset adapters
│   ├── landmarks/                 # Extraction, normalization, sequences
│   ├── models/                    # Pydantic schemas + SQLite database
│   ├── recognition/               # Hybrid recognition, debouncing, HELP
│   └── services/
│       ├── gemini_service.py      # Gemini multimodal/AI support
│       └── tts_service.py
│
├── data/
│   ├── asl/
│   ├── isl/
│   ├── bsl/
│   ├── custom/
│   └── sign_system.db             # SQLite custom-sign database
│
├── tests/                          # Automated test suite
└── README.md

🔬 Landmark Representation

Each detected frame is represented using:

21 hand landmarks

3 coordinates per landmark: x, y, z

63 features per hand

2 hands

126 features per frame

Dynamic signs are represented as:

30 frames × 126 features = 3,780 features

The sequence layer can:

Validate landmark dimensions

Normalize landmark values

Center and scale hands

Resample sequences to 30 frames

Calculate motion energy

Build static/dynamic dataset samples

This gives the recognition layer a consistent representation across
camera frames and datasets.

🎯 Recognition Pipeline

The central Sign2VoicePipeline follows this flow:

Raw ML Prediction
       ↓
MLAdapter.normalize()
       ↓
GestureProcessor
       ├── confidence gate
       ├── stability window
       ├── cooldown
       └── duplicate suppression
       ↓
SentenceBuilder
       ↓
AppState

Default Sign2Voice pipeline thresholds:

Setting                                    Default

ML confidence threshold                     0.75
Stability window            3 consecutive frames
Gesture cooldown                           1.5 s
TTS language                                  en
OpenAI model                         gpt-4o-mini

The newer src/ recognition layer has its own configurable recognition
thresholds, including motion-energy and HELP persistence settings.

✨ AI Sentence Refinement

AI is deliberately not called on every video frame.

The raw gesture sequence remains the source of truth. AI is only used as
an enhancement layer after the user requests sentence improvement.

Example:

Raw gesture sequence:
hello i need water

AI-refined output:
Hello, I need water.

The sentence corrector is instructed to preserve meaning and only
improve grammar, capitalization, and punctuation.

The project also includes a Gemini service for:

Sign-language recognition assistance

Natural-language sentence polishing

Sign descriptions

Multimodal landmark/image analysis

Offline geometric/kinematic fallback reasoning

If an AI API is unavailable, the application falls back to the
original/raw sentence rather than failing the interaction.

🔊 Voice Output

Voice output uses two complementary browser/application paths:

gTTS

The backend TTSEngine converts text to MP3 bytes and exposes them to
the Streamlit interface.

Browser Speech Synthesis

The Streamlit UI can also invoke the browser's built-in
SpeechSynthesis API for immediate voice playback.

This gives the hackathon demo a fast voice-feedback path while retaining
a dedicated TTS service.

🧩 Custom Sign Storage

Personalized signs are persisted in SQLite.

The database contains:

custom_signs
    │
    └── custom_samples

A custom sign stores:

Sign ID

User ID

Label

Description

Sample count

Static/dynamic sample information

Creation/update timestamps

Samples store either:

126-dimensional static features, or

30-frame dynamic landmark sequences

🌍 Included Dataset Samples

The repository includes landmark-based JSONL samples:

Language              Static Samples      Dynamic Samples Example Labels

ASL                               10                   10 Hello, Thank
You, Yes, No,
Peace, Please,
Sorry, Help,
Friend, Book

ISL                               25                   25 Namaste, Water,
Good, Bad,
Victory, Dance,
School, Home,
Family, Play

These are lightweight local samples intended for the project/demo
pipeline. The repository also provides adapters for external dataset
ingestion.

📥 Dataset Ingestion

The ingestion layer provides adapters for:

ASL

ISL

BSL

The API can ingest supported CSV/JSON sources and exposes dataset status
information such as:

Total samples

Static/dynamic counts

Distinct labels

Static classes

Dynamic classes

Last update information

Synthetic data generation is also available for development and testing.

🔌 FastAPI Endpoints

Start the API with:

python run_server.py

Then open:

http://127.0.0.1:8000

Interactive documentation:

http://127.0.0.1:8000/docs

Health

GET /health

Recognition

POST /api/recognition/frame
POST /api/recognition/sequence
GET  /api/recognition/transcript
POST /api/recognition/transcript/clear
POST /api/recognition/transcript/pause
POST /api/recognition/transcript/resume
POST /api/recognition/help/reset
WS   /api/recognition/ws

The WebSocket supports landmark or image messages and streams
recognition events such as transcript updates and HELP detection.

Custom Signs

POST   /api/custom-sign/start
POST   /api/custom-sign/sample
POST   /api/custom-sign/save
GET    /api/custom-signs
GET    /api/custom-sign/{sign_id}
PUT    /api/custom-sign/{sign_id}
POST   /api/custom-sign/{sign_id}/samples
DELETE /api/custom-sign/{sign_id}

Datasets

GET  /api/datasets/status
GET  /api/datasets/{language}/status
POST /api/datasets/generate-synthetic
POST /api/datasets/ingest

Gemini Support

GET  /api/gemini/status
POST /api/gemini/polish-sentence
POST /api/gemini/describe-sign

⚙️ Installation

Requirements

Python 3.10+

Webcam for live recognition

Internet connection for OpenAI/gTTS/Gemini features

A compatible environment for MediaPipe

1. Clone the repository

git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_REPOSITORY_FOLDER>

2. Create a virtual environment

Windows:

python -m venv .venv
.venv\Scripts\activate

macOS/Linux:

python3 -m venv .venv
source .venv/bin/activate

3. Install dependencies

pip install -r requirements.txt

4. Configure environment variables

Copy:

.env.example → .env

Then add the API keys you want to use.

Example:

OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4o-mini
OPENAI_TIMEOUT=10.0

GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL_NAME=gemini-2.5-flash

CONFIDENCE_THRESHOLD=0.75
STABILITY_WINDOW=3
GESTURE_COOLDOWN=1.5

TTS_LANGUAGE=en
TTS_SLOW=false

DEMO_MODE=false

Never commit .env or API keys to GitHub.

▶️ Run the Hackathon Demo

Streamlit UI

streamlit run app.py

The interface provides:

Live Sign to Voice

Custom Sign Studio

Multilingual Vocabulary

Language selection

Live/snapshot camera analysis

Quick gesture triggers

Demo sequence

Sentence improvement

Voice output

Custom sign management

app.py also contains a small auto-bootstrap mechanism, so running:

python app.py

can launch Streamlit automatically.

FastAPI service

In another terminal:

python run_server.py

For development with reload:

python run_server.py --reload

🎬 Demo Mode

For a reliable hackathon presentation, enable:

DEMO_MODE=true

The built-in demo injects a stable gesture sequence so the sentence
pipeline can be demonstrated without depending entirely on camera
conditions or a trained local classifier.

This is especially useful for:

Stage demos

Judging sessions

Offline environments

Automated tests

Reproducible presentations

🧪 Testing

The repository contains automated tests covering:

Gesture stability and cooldown

Duplicate suppression

Sentence building

Application state

TTS behavior

Landmark extraction and normalization

Sequence buffering/resampling

Dataset adapters

Custom signs

Recognition engine

HELP detection

Gemini fallback/error handling

FastAPI endpoints

End-to-end pipeline behavior

Run:

pytest -v

The repository currently contains 76 test functions across the test
modules.

Test count describes the checked-in test suite; actual pass/fail
status depends on the environment and installed dependencies.

🔐 Privacy & Security Notes

API keys are read from environment variables.

.env should never be committed.

Custom signs are stored locally in SQLite.

Camera/landmark processing is performed by the application pipeline.

External AI services are optional and should only be enabled when
their privacy/usage terms are acceptable for your deployment.

⚠️ Current Recognition / Model Notes

The repository contains the MediaPipe hand-landmark model and
lightweight sample datasets.

The hybrid recognition engine is designed to load optional local model
bundles:

static_gesture_model.pkl
dynamic_gesture_model.pkl

If those trained model files are not present, the engine uses its
fallback recognition path rather than crashing.

The Streamlit demo UI also contains a lightweight landmark/handshape
classifier and custom-sign matching path so the application can remain
demonstrable without requiring a separately trained production
classifier.

For a production deployment, replace/demo-bypass recognition logic
should be replaced or augmented with a properly trained and validated
sign-language model using sufficiently representative data.

🛠️ Technology Stack

Layer                    Technology

UI                       Streamlit
API                      FastAPI
Server                   Uvicorn
Computer Vision          OpenCV
Hand Tracking            MediaPipe
Numerical Processing     NumPy
ML Model Loading         scikit-learn + joblib
AI Sentence Refinement   OpenAI
Multimodal AI Support    Google Gemini
Text-to-Speech           gTTS + Browser SpeechSynthesis
Persistence              SQLite
Validation               Pydantic
Testing                  Pytest

💡 Hackathon Highlights

♿ Accessibility-first design

The system is built around a real communication problem rather than a
generic AI demo.

⚡ Real-time interaction

Gesture processing is designed around continuous camera input, temporal
stability, and low-latency state updates.

🧠 Hybrid intelligence

The architecture can combine deterministic landmark processing, local ML
models, heuristic fallback recognition, custom vocabulary, and
multimodal Gemini assistance.

✍️ Personalization

Users can teach the system signs that matter specifically to them.

🌍 Multilingual foundation

The data/API architecture is not locked to a single sign language.

🆘 Safety-aware recognition

Persistent HELP detection is treated as a separate event layer rather
than simply another transcript word.

🧪 Demo-friendly engineering

Demo mode, fallbacks, structured APIs, and automated tests make the
project easier to present and evaluate.

🗺️ Future Roadmap

Potential next steps for a production-grade system:

Train and bundle validated ASL/ISL/BSL static and dynamic
classifiers

Expand vocabulary substantially

Add signer-independent evaluation and accuracy metrics

Improve two-hand interaction recognition

Add sentence context and conversation memory with privacy controls

Add richer meeting/captioning workflows

Add mobile-friendly deployment

Add multilingual spoken-language output

Add confidence visualization and recognition explanations

Improve personalized-sign matching with learned embeddings

Add model benchmarking and latency dashboards

🤝 Contributing

Contributions are welcome.

A typical workflow:

git checkout -b feature/your-feature
# make your changes
pytest -v
git add .
git commit -m "Add: your feature"
git push origin feature/your-feature

Then open a pull request.

📄 License

Add the project's chosen license here before publishing the repository.

For a hackathon submission, we recommend explicitly choosing a license
such as MIT if that matches your team's intent and the licenses of any
third-party assets you redistribute.

🏆 Built for Hackathon Impact

VoiceSignAI is designed to turn sign-language gestures into a
practical communication interface --- combining computer vision, AI,
personalization, multilingual support, and voice output in one
accessible system.

From hands to words. From words to voice. From barriers to
conversation. 🤟
