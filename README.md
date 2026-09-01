# 🤟 VoiceSignAI

### **Turning Signs Into Speech. Breaking Barriers Through AI.**

> **VoiceSignAI** is an AI-powered assistive communication platform that understands sign-language gestures through a camera, converts them into meaningful text, improves the sentence using Generative AI, and transforms the final message into speech.

<p align="center">

**👋 Sign Language → 👁️ Computer Vision → 🧠 Recognition → ✍️ AI → 🔊 Voice**

</p>

---

## 🚀 About VoiceSignAI

Communication should never depend on whether two people speak the same language — and that includes sign language.

Millions of people communicate through sign language, while many people around them may not understand it. This creates a communication barrier in everyday situations such as education, healthcare, public services, workplaces, and social interaction.

**VoiceSignAI** is built to reduce that barrier.

The system uses a camera to observe hand gestures, extracts hand landmarks using computer vision, identifies gestures, builds them into a sentence, optionally improves the sentence with Generative AI, and produces spoken audio.

Instead of treating sign recognition as simply:

```text
Gesture → Word
```

VoiceSignAI builds a complete communication pipeline:

```text
Gesture
   ↓
Hand Landmark Detection
   ↓
Gesture Recognition
   ↓
Confidence & Stability Filtering
   ↓
Sentence Construction
   ↓
AI-Powered Language Refinement
   ↓
Text-to-Speech
   ↓
Voice
```

The result is an accessibility-focused system designed to make sign-based communication easier to understand for people who may not know sign language.

---

# 🎯 Problem Statement

## The Communication Gap

A person who uses sign language can communicate naturally with another signer. However, communication becomes difficult when the other person does not understand sign language.

Existing solutions often focus primarily on recognizing isolated gestures. Real communication is more than recognizing individual hand shapes.

A practical communication assistant needs to handle:

* Real-time hand tracking
* Static and dynamic gestures
* Multiple sign-language modes
* Noisy frame-by-frame predictions
* Repeated gesture detection
* Sentence construction
* Natural language correction
* Speech output
* Personalized signs
* Emergency communication

### Our approach

**VoiceSignAI transforms visual sign-language input into a complete spoken communication pipeline.**

The goal is not simply to recognize a hand pose.

The goal is:

> **Understand the sign → construct the message → make the message natural → give it a voice.**

---

# 💡 What Makes VoiceSignAI Different?

VoiceSignAI combines several layers into one system instead of treating gesture recognition as an isolated machine-learning problem.

### 🧠 Hybrid Recognition Architecture

The recognition backend is designed to work with:

* Static gesture models
* Dynamic gesture models
* Motion analysis
* Landmark-based recognition
* Custom user signs
* Fallback recognition

This allows the system to distinguish between gestures that are primarily based on **hand shape** and gestures where **movement over time** matters.

---

### ✍️ Teach My Sign

Everyone may not perform a sign in exactly the same way.

VoiceSignAI therefore includes a **Custom Sign Studio** where users can teach the system their own signs.

Users can create personalized vocabulary using:

* Static hand poses
* Dynamic motion sequences
* Multiple training samples
* Landmark normalization
* SQLite persistence

This makes the system adaptable rather than being restricted to a completely fixed vocabulary.

---

### 🤖 AI Is Used Where It Matters

The Generative AI layer does not replace gesture recognition.

Instead:

```text
Computer Vision
       ↓
Recognized sign words
       ↓
Generative AI
       ↓
Natural sentence
```

For example, the recognition system may produce:

```text
hello i need water
```

The AI layer can turn that into a more natural sentence:

```text
Hello, I need water.
```

This separation keeps the recognition pipeline understandable while using AI where it provides the most value: **language refinement and assistance**.

---

### 🚨 Help & Emergency Awareness

VoiceSignAI also contains a dedicated help-detection mechanism.

The system monitors recognized labels associated with assistance or emergency situations, including concepts such as:

```text
help
emergency
assistance
need help
danger
sos
```

The detector requires the signal to persist for a configurable period rather than triggering immediately from a single noisy prediction.

This creates a foundation for accessibility-oriented emergency communication.

---

# 🌍 Multilingual Sign-Language Architecture

VoiceSignAI provides separate language modes for:

| Mode          | Description                     |
| ------------- | ------------------------------- |
| 🇺🇸 **ASL**  | American Sign Language          |
| 🇮🇳 **ISL**  | Indian Sign Language            |
| 🇬🇧 **BSL**  | British Sign Language           |
| ✍️ **CUSTOM** | User-defined personalized signs |

The data architecture maintains separate directories and ingestion pipelines for ASL, ISL and BSL.

The recognition layer also carries the selected language throughout the recognition pipeline.

---

# 🧩 System Architecture

```text
                         ┌─────────────────────┐
                         │       CAMERA        │
                         │    Webcam Input     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │     MediaPipe       │
                         │  Hand Landmarkers   │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Landmark Processing │
                         │ Normalization       │
                         │ Validation          │
                         └──────────┬──────────┘
                                    │
                                    ▼
                    ┌──────────────────────────────┐
                    │    Recognition Engine        │
                    │                              │
                    │ • Static Recognition         │
                    │ • Dynamic Recognition        │
                    │ • Motion Analysis            │
                    │ • Custom Signs               │
                    │ • Fallback Recognition       │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │ Confidence & Stability Layer │
                    │                              │
                    │ • Confidence threshold       │
                    │ • Debouncing                 │
                    │ • Duplicate suppression      │
                    │ • Cooldown                   │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                         ┌─────────────────────┐
                         │  Sentence Builder   │
                         │                     │
                         │ Raw Sign Sequence  │
                         └──────────┬──────────┘
                                    │
                         ┌──────────┴──────────┐
                         │                     │
                         ▼                     ▼
                ┌─────────────────┐   ┌─────────────────┐
                │ Generative AI   │   │ Text-to-Speech  │
                │                 │   │                 │
                │ OpenAI / Gemini │   │      gTTS       │
                └────────┬────────┘   └────────┬────────┘
                         │                     │
                         ▼                     ▼
                ┌────────────────────────────────────┐
                │          Spoken Communication      │
                └────────────────────────────────────┘
```

---

# 👁️ Computer Vision Pipeline

The first layer of VoiceSignAI is visual perception.

The system uses **MediaPipe Hand Landmarker** to identify hand landmarks from camera frames.

Each detected hand contains:

```text
21 landmarks
×
3 coordinates (X, Y, Z)
```

Therefore:

```text
21 × 3 = 63 features per hand
```

With support for two hands:

```text
63 × 2 = 126 features per frame
```

These landmark vectors become the foundation for gesture recognition.

---

# 🧠 Static & Dynamic Recognition

Not every sign is represented in the same way.

Some signs can be identified primarily from a single hand configuration.

Others depend on movement.

VoiceSignAI therefore separates recognition into two conceptual categories.

## Static gestures

Static gestures are represented using the spatial configuration of hand landmarks.

The system analyzes characteristics such as:

* Finger extension
* Finger positions
* Wrist position
* Relative distances
* Hand shape
* Finger relationships

---

## Dynamic gestures

Dynamic signs require information across multiple frames.

VoiceSignAI represents dynamic sequences using:

```text
30 frames
×
126 features per frame
=
3,780 features
```

The system can analyze motion characteristics such as:

* Motion energy
* Dominant hand
* Wrist trajectory
* Path length
* Net displacement
* Direction changes
* Circularity
* Two-hand activity
* Relative wrist movement

This gives the recognition architecture temporal information rather than relying solely on a single frame.

---

# 📐 Landmark Normalization

Raw camera coordinates can vary depending on:

* Hand position
* Camera position
* Scale
* Distance from camera

VoiceSignAI therefore contains a landmark normalization layer.

The processing pipeline can:

* Validate landmark dimensions
* Detect invalid values
* Handle NaN/Inf values
* Center hand coordinates around the wrist
* Scale landmark representations
* Normalize temporal sequences

This creates a more consistent representation for recognition.

---

# 🛡️ Recognition Stability

A major challenge with real-time gesture recognition is that the same gesture may be detected across many consecutive frames.

Without filtering, one gesture could become:

```text
hello hello hello hello hello
```

instead of:

```text
hello
```

VoiceSignAI addresses this using a recognition stability layer.

```text
Raw Prediction
      ↓
Confidence Check
      ↓
Repeated Prediction Validation
      ↓
Debouncing
      ↓
Cooldown
      ↓
Duplicate Suppression
      ↓
Accepted Word
```

This helps prevent noisy frame-level predictions from producing an unreadable sentence.

---

# ✍️ Teach My Sign

## Personalized Sign Recognition

One of the most important parts of VoiceSignAI is the **Custom Sign Studio**.

Instead of forcing every user to depend entirely on a predefined vocabulary, the system allows personalized signs to be stored.

A custom sign can contain:

* Label
* Description
* User identifier
* Static samples
* Dynamic samples
* Normalized landmarks
* Motion information

The system requires a minimum of **3 samples** when creating a custom sign.

The recommended number of samples is **5**.

---

## Static Custom Signs

A static sign can be captured as a landmark feature vector.

The system:

```text
Capture
   ↓
Extract landmarks
   ↓
Normalize
   ↓
Store
   ↓
Match against future gestures
```

---

## Dynamic Custom Signs

Dynamic signs can be captured as a sequence of frames.

The system:

```text
Motion Clip
    ↓
Landmark Extraction
    ↓
Sequence Resampling
    ↓
Normalization
    ↓
Motion Analysis
    ↓
SQLite Storage
```

Dynamic sequences are resampled to a standard length of:

```text
30 frames
```

This provides a consistent representation for comparison.

---

# 🗃️ Dataset Architecture

VoiceSignAI contains a dataset ingestion layer for:

```text
ASL
ISL
BSL
```

The ingestion system supports different dataset formats and maintains language-specific storage.

The backend includes adapters for:

* ASL dataset ingestion
* ISL dataset ingestion
* BSL dataset ingestion
* CSV-based sources
* JSON-based sources
* Synthetic dataset generation

Dataset status can also be queried through the API.

---

# 🤖 Generative AI Layer

VoiceSignAI integrates Generative AI as an **assistive language layer**.

## OpenAI

The Streamlit pipeline contains an OpenAI-powered sentence correction service.

The model is used to improve:

* Grammar
* Punctuation
* Sentence structure
* Natural phrasing

The important design principle is:

> **The AI does not decide what gesture was performed.**

The recognition system produces the words first.

The AI then improves the resulting language.

---

## Google Gemini

The backend also contains a Gemini service layer.

Gemini can be used for:

* Sentence polishing
* Sign descriptions
* Assistive sign explanations
* Conversational text refinement
* Landmark/trajectory-assisted interpretation

The backend exposes dedicated Gemini API endpoints.

---

# 🔊 Text-to-Speech

After the sentence is constructed, VoiceSignAI can turn it into speech.

The project contains a TTS service based on:

**Google Text-to-Speech (gTTS)**

The flow is:

```text
Recognized Signs
      ↓
Raw Sentence
      ↓
AI Enhancement
      ↓
Final Sentence
      ↓
gTTS
      ↓
MP3 Audio
      ↓
Browser Playback
```

The Streamlit application can also use browser speech capabilities for immediate voice interaction.

---

# 🚨 Help Detection

VoiceSignAI includes a dedicated `HelpDetector`.

Instead of treating every detection of a help-related sign as an emergency event, the system tracks how long the signal remains active.

The configured persistence period is:

```text
5 seconds
```

The conceptual flow is:

```text
HELP detected
      ↓
Start timer
      ↓
Continue monitoring
      ↓
Signal persists
      ↓
HELP_DETECTED event
```

This reduces the possibility of a single noisy prediction immediately producing an emergency event.

---

# 🖥️ User Interface

The main interface is built with **Streamlit**.

The application provides:

### 🔍 Lens Mode

The primary sign-recognition experience.

It provides:

* Language selection
* Camera interaction
* Live hand landmark visualization
* Current gesture information
* Sentence construction
* AI sentence improvement
* Text-to-speech
* Undo
* Clear
* Reset
* Demo controls

---

### ✍️ Custom Sign Studio

The personalized vocabulary interface provides:

* Static sign enrollment
* Dynamic sign recording
* Dynamic video upload
* Custom sign storage
* Custom sign management
* Sample management
* Voice testing

---

### 📊 Dataset Interface

The application also exposes dataset-related information and management functionality.

This provides visibility into the multilingual data layer supporting the recognition system.

---

# 🎬 Demo Mode

VoiceSignAI includes a built-in demo mechanism.

This is useful for:

* Hackathon demonstrations
* Development without a trained model
* Testing the sentence pipeline
* Demonstrating AI refinement
* Demonstrating speech output
* Environments where camera input is unavailable

The application can inject a scripted sequence of gestures and demonstrate the complete:

```text
Recognition
→ Sentence
→ AI
→ Voice
```

workflow.

---

# ⚙️ Backend Architecture

In addition to the Streamlit interface, VoiceSignAI contains a modular **FastAPI backend**.

The backend provides REST and WebSocket-based infrastructure for:

* Recognition
* Transcript management
* Custom signs
* Dataset management
* Gemini services
* Real-time recognition events

This architecture makes the project extensible beyond the current Streamlit interface.

---

# 🔌 API Structure

The backend is organized into several functional areas.

## Recognition

```text
POST /api/recognition/frame
POST /api/recognition/sequence
GET  /api/recognition/transcript

POST /api/recognition/transcript/clear
POST /api/recognition/transcript/pause
POST /api/recognition/transcript/resume

POST /api/recognition/help/reset

WS /api/recognition/ws
```

---

## Custom Signs

```text
POST   /api/custom-sign/start
POST   /api/custom-sign/sample
POST   /api/custom-sign/save

GET    /api/custom-signs
GET    /api/custom-sign/{sign_id}

PUT    /api/custom-sign/{sign_id}

POST   /api/custom-sign/{sign_id}/samples

DELETE /api/custom-sign/{sign_id}
```

---

## Dataset Management

```text
GET  /api/datasets/status
GET  /api/datasets/{language}/status

POST /api/datasets/generate-synthetic
POST /api/datasets/ingest
```

---

## Gemini

```text
GET  /api/gemini/status

POST /api/gemini/polish-sentence
POST /api/gemini/describe-sign
```

---

# 🗄️ Data Storage

VoiceSignAI uses **SQLite** for persistent custom-sign data.

The database stores information required for:

* Custom sign definitions
* User associations
* Sign descriptions
* Static samples
* Dynamic samples
* Landmark representations
* Sample metadata

This allows personalized signs to remain available across application sessions.

---

# 📁 Project Structure

```text
VoiceSignAI/
│
├── app.py
├── build_ui.py
├── config.py
├── run_server.py
├── requirements.txt
├── .env.example
├── .gitignore
├── hand_landmarker.task
│
├── ai/
│   └── sentence_corrector.py
│
├── backend/
│   ├── gesture_processor.py
│   ├── sentence_builder.py
│   └── state_manager.py
│
├── integration/
│   ├── ml_adapter.py
│   └── pipeline.py
│
├── speech/
│   └── tts.py
│
├── ml/
│
├── data/
│   ├── asl/
│   ├── isl/
│   ├── bsl/
│   ├── custom/
│   └── sign_system.db
│
├── src/
│   │
│   ├── api/
│   │   ├── app.py
│   │   ├── routes_custom.py
│   │   ├── routes_dataset.py
│   │   ├── routes_gemini.py
│   │   └── routes_recognition.py
│   │
│   ├── custom_signs/
│   │   ├── manager.py
│   │   └── service.py
│   │
│   ├── ingestion/
│   │   ├── asl_adapter.py
│   │   ├── bsl_adapter.py
│   │   ├── isl_adapter.py
│   │   ├── base_adapter.py
│   │   ├── dataset_manager.py
│   │   └── synthetic_data.py
│   │
│   ├── landmarks/
│   │   ├── extractor.py
│   │   ├── normalizer.py
│   │   └── sequence.py
│   │
│   ├── models/
│   │   ├── database.py
│   │   └── schemas.py
│   │
│   ├── recognition/
│   │   ├── debouncer.py
│   │   ├── help_detector.py
│   │   ├── hybrid_engine.py
│   │   ├── inference.py
│   │   ├── interface.py
│   │   └── mock_recognizer.py
│   │
│   └── services/
│       ├── gemini_service.py
│       └── tts_service.py
│
└── tests/
    ├── test_api.py
    ├── test_custom_signs.py
    ├── test_datasets.py
    ├── test_gemini.py
    ├── test_gesture_processor.py
    ├── test_landmarks.py
    ├── test_pipeline.py
    ├── test_recognition.py
    ├── test_sentence_builder.py
    ├── test_state_manager.py
    └── test_tts.py
```

---

# 🛠️ Technology Stack

| Layer                      | Technology                |
| -------------------------- | ------------------------- |
| 🎨 Application UI          | Streamlit                 |
| 👁️ Computer Vision        | MediaPipe, OpenCV         |
| ✋ Hand Tracking            | MediaPipe Hand Landmarker |
| 🔢 Numerical Processing    | NumPy                     |
| 🧠 Machine Learning        | scikit-learn, joblib      |
| 🤖 Generative AI           | OpenAI, Google Gemini     |
| 🔊 Text-to-Speech          | gTTS                      |
| ⚡ Backend API              | FastAPI                   |
| 🔄 Real-Time Communication | WebSockets                |
| 🗃️ Database               | SQLite                    |
| 📊 Dataset Processing      | JSONL, CSV, JSON          |
| 🧪 Testing                 | pytest                    |
| ⚙️ Configuration           | python-dotenv             |

---

# 📐 Technical Overview

VoiceSignAI's landmark representation is built around:

```text
21 landmarks / hand
×
3 coordinates / landmark
=
63 features / hand
```

For two hands:

```text
63 × 2
=
126 features / frame
```

For dynamic sequences:

```text
126 features / frame
×
30 frames
=
3,780 features / sequence
```

The system uses these representations for landmark processing, motion analysis, custom-sign storage, and model-based recognition.

---

# 🔄 End-to-End Workflow

A complete interaction can look like this:

### 1️⃣ Capture

The user presents a sign to the camera.

### 2️⃣ Detect

MediaPipe detects the hand and extracts its landmarks.

### 3️⃣ Normalize

The landmark representation is validated and normalized.

### 4️⃣ Analyze

The system determines spatial and/or temporal characteristics.

### 5️⃣ Recognize

The recognition layer identifies the most appropriate gesture.

### 6️⃣ Stabilize

Confidence filtering and debouncing prevent accidental duplicate words.

### 7️⃣ Build

Accepted signs are accumulated into a sentence.

### 8️⃣ Refine

Generative AI can improve grammar and natural phrasing.

### 9️⃣ Speak

The final sentence is converted into speech.

### 🔟 Communicate

The listener hears the message.

```text
          🤟
       SIGN
         ↓
       👁️
   DETECTION
         ↓
       🧠
   RECOGNITION
         ↓
       📝
    SENTENCE
         ↓
       ✨
       AI
         ↓
       🔊
      VOICE
```

---

# 🔐 Configuration

VoiceSignAI supports environment-based configuration.

Create a `.env` file based on:

```text
.env.example
```

For OpenAI sentence enhancement:

```env
OPENAI_API_KEY=your_key_here
```

For Gemini services:

```env
GEMINI_API_KEY=your_key_here
```

Optional configuration values include:

```env
OPENAI_MODEL=gpt-4o-mini
OPENAI_TIMEOUT=10.0

GEMINI_MODEL_NAME=gemini-2.5-flash

CONFIDENCE_THRESHOLD=0.65
```

API keys should always remain private and should never be committed to the repository.

---

# ▶️ Running VoiceSignAI

## 1. Clone the repository

```bash
git clone https://github.com/YuvaHQ/VoiceSign-AI.git
cd VoiceSign-AI
```

---

## 2. Create a virtual environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Configure environment variables

Create a `.env` file from `.env.example`.

Add the AI credentials you want to enable.

The application is designed to degrade gracefully when optional AI services are unavailable, allowing the raw recognized sentence to remain usable.

---

## 5. Start the application

```bash
streamlit run app.py
```

The application will start the VoiceSignAI interface in your browser.

---

# 🌐 Starting the API Backend

The FastAPI backend can be started with:

```bash
python run_server.py
```

Or:

```bash
python run_server.py --host 127.0.0.1 --port 8000 --reload
```

Once running, FastAPI's interactive API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

---

# 🧪 Testing

VoiceSignAI includes an automated test suite covering major system components.

The tests cover areas including:

* API endpoints
* Custom signs
* Dataset management
* Gemini services
* Gesture processing
* Landmark processing
* Recognition
* Sentence construction
* Application state
* TTS

Run:

```bash
pytest -v
```

---

# 🧪 Development Philosophy

VoiceSignAI was designed with modularity in mind.

Instead of placing everything inside one large recognition function, the project separates:

```text
Vision
│
├── Landmark Extraction
├── Landmark Normalization
└── Sequence Processing
        │
        ▼
Recognition
│
├── Static Recognition
├── Dynamic Recognition
├── Custom Recognition
└── Fallback Recognition
        │
        ▼
Application Logic
│
├── Debouncing
├── Sentence Building
├── State Management
└── Help Detection
        │
        ▼
AI & Speech
│
├── OpenAI
├── Gemini
└── gTTS
```

This makes it easier to replace or improve individual components without rebuilding the entire application.

---

# 🧠 Why Separate Recognition From AI?

A key architectural decision in VoiceSignAI is keeping gesture recognition and language generation separate.

The system first determines:

```text
"What sign was detected?"
```

Only after that does AI help answer:

```text
"How can these recognized words be expressed naturally?"
```

This distinction is important because a language model should not be allowed to freely invent the meaning of a gesture.

The raw sentence therefore remains separate from the AI-enhanced sentence.

```text
RAW RECOGNITION
hello i need water

        ↓

AI REFINEMENT

Hello, I need water.
```

The original recognition output remains available as the underlying source.

---

# ⚡ Graceful Fallbacks

VoiceSignAI is designed so optional services do not have to bring down the entire application.

For example:

```text
AI Available
     ↓
AI-enhanced sentence
```

or:

```text
AI Unavailable
     ↓
Raw recognized sentence
```

Similarly, recognition can fall back when trained model assets are unavailable, and demo functionality can be used for presentations and development.

This makes the application more practical for different environments.

---

# 🎯 Hackathon Demonstration Flow

A simple demonstration of VoiceSignAI can showcase the complete system in a few steps.

### Step 1 — Open Lens Mode

Start the application and select the required sign-language mode.

### Step 2 — Show a Gesture

Present a gesture to the camera.

### Step 3 — Recognition

The system detects the hand landmarks and identifies the gesture.

### Step 4 — Sentence Construction

Continue presenting signs and watch them accumulate into a sentence.

### Step 5 — AI Enhancement

Use the AI improvement functionality to transform fragmented recognized words into a natural sentence.

### Step 6 — Voice Output

Use the speech function to hear the generated sentence.

### Step 7 — Teach My Sign

Open Custom Sign Studio and demonstrate how a personalized sign can be recorded and stored.

### Step 8 — Emergency Detection

Demonstrate the persistent HELP detection architecture.

### Step 9 — Backend

Open the FastAPI documentation to demonstrate the modular backend and API architecture.

---

# 🏆 Hackathon Vision

VoiceSignAI is built around a simple but powerful idea:

> **Communication should be accessible to everyone.**

A sign is not merely a gesture.

It can represent:

* A greeting
* A request
* A need
* A conversation
* An emergency
* A complete thought

VoiceSignAI attempts to carry that meaning across the communication gap.

```text
🤟
SIGN
  ↓
👁️
SEE
  ↓
🧠
UNDERSTAND
  ↓
✍️
REFINE
  ↓
🔊
SPEAK
```

---

# 🔮 Future Scope

The current architecture provides a foundation for further development.

Potential improvements include:

* Larger real-world sign-language datasets
* More comprehensive ISL vocabulary
* More comprehensive ASL and BSL vocabulary
* Improved signer-independent recognition
* More advanced temporal gesture models
* Better two-hand gesture recognition
* Offline AI and speech capabilities
* Mobile deployment
* Edge-device inference
* User profiles
* Richer conversational sessions
* Real-time multi-user communication
* Recognition accuracy benchmarking
* Latency benchmarking
* Expanded accessibility features

---

# 🌟 The Bigger Picture

VoiceSignAI is more than a gesture classifier.

It combines:

**Computer Vision**

to see the hands,

**Machine Learning**

to understand gesture patterns,

**Generative AI**

to make recognized language natural,

**Text-to-Speech**

to turn that language into a voice,

and

**Personalization**

to adapt the system to individual users.

The ultimate pipeline is simple:

# **🤟 Sign → 🧠 Understand → ✨ Refine → 🔊 Speak**

---

<p align="center">

## 🤟 VoiceSignAI

### **Let every sign have a voice.**

**Built by Sloppy Minds for Hackverse: Into the Web**

</p>
