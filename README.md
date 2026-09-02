# 🤟 VoiceSignAI

### **Turning Signs Into Speech. Breaking Barriers Through AI.**

<p align="center">

**👋 Sign Language → 👁️ Computer Vision → 🧠 Recognition → ✨ AI → 🔊 Voice**

</p>

<p align="center">

### Built by **Sloppy Minds** for **Hackverse: Into the Web**

</p>

---

## 🌟 What is VoiceSignAI?

**VoiceSignAI** is an AI-powered assistive communication platform that understands sign-language gestures through a camera, converts them into meaningful text, improves the sentence using Generative AI, and transforms the final message into speech.

The project is built around a simple idea:

> **Communication should not be limited by whether the person listening understands sign language.**

VoiceSignAI combines **computer vision, gesture recognition, personalization, Generative AI, and text-to-speech** into one communication pipeline.

Instead of simply recognizing an isolated hand gesture, the system aims to turn a user's signs into an understandable spoken message.

```text
🤟 Sign
   ↓
👁️ Hand Landmark Detection
   ↓
🧠 Gesture Recognition
   ↓
🛡️ Confidence & Stability Filtering
   ↓
📝 Sentence Construction
   ↓
✨ AI Language Refinement
   ↓
🔊 Text-to-Speech
   ↓
🗣️ Spoken Communication
```

---

# 🎯 The Problem

Communication becomes difficult when a person who uses sign language interacts with someone who does not understand it.

The challenge becomes even more complex when we consider that **not every person can physically perform a conventional sign in exactly the same way**.

People may have different:

* Finger mobility
* Hand mobility
* Range of motion
* Comfortable hand positions
* Natural movement patterns

A rigid sign-recognition system can unintentionally create another barrier by expecting every user to reproduce a predefined gesture in exactly the same configuration.

### The traditional approach

```text
Predefined Sign
      ↓
User must reproduce it exactly
      ↓
Recognition
```

### Our approach

```text
User's Natural Gesture
      ↓
VoiceSignAI learns it
      ↓
User-defined Meaning
      ↓
Personalized Recognition
      ↓
Communication
```

**VoiceSignAI is designed to adapt to the user instead of forcing the user to adapt to the technology.**

---

# 💡 Our Solution

VoiceSignAI creates an end-to-end communication system where a user can:

1. Perform a sign using a camera.
2. Extract hand landmarks using computer vision.
3. Recognize static or dynamic gestures.
4. Filter unstable predictions.
5. Build recognized signs into a sentence.
6. Improve the sentence using Generative AI.
7. Convert the final sentence into speech.

And most importantly:

### **Users can teach VoiceSignAI their own signs.**

If a user cannot comfortably reproduce a predefined sign because of differences in finger or hand movement, they can use **Teach My Sign** to create a personalized gesture and associate it with their own word or meaning.

This makes personalization a core **accessibility feature**, not simply an extra feature.

---

# ♿ Adaptive Accessibility — Your Sign, Your Way

## **Technology should adapt to the person — not force the person to adapt to the technology.**

This is one of the core ideas behind VoiceSignAI.

A predefined sign may require a particular finger position, hand shape, or movement that is difficult for some users to perform.

With **Teach My Sign**, users can provide a gesture that is natural and comfortable for them and assign a meaning to it.

For example:

```text
User's Comfortable Gesture
          ↓
       Capture
          ↓
  Extract Landmarks
          ↓
      Normalize
          ↓
User assigns a meaning
          ↓
       "Water"
          ↓
     Save the Sign
          ↓
Recognize it in future
```

The user effectively tells the system:

> **"This is how I perform this sign, and this is what it means."**

VoiceSignAI can then associate that personalized gesture with the selected meaning.

---

## 🧩 Why Personalized Signs Matter

Imagine a user who has limited movement in one or more fingers.

A conventional sign may require a hand configuration that is uncomfortable or impossible for that person to reproduce.

Instead of requiring the user to imitate the predefined gesture, VoiceSignAI allows them to:

```text
Perform a comfortable gesture
             ↓
       Teach the system
             ↓
      Assign a meaning
             ↓
       Save the sign
             ↓
      Use it naturally
```

This allows the communication vocabulary to become **personalized around the individual user**.

The system can learn the user's own representation of a concept.

---

# ✍️ Teach My Sign

## **A personalized vocabulary system built for adaptability**

VoiceSignAI includes a **Custom Sign Studio** that allows users to create their own signs.

A custom sign can be associated with:

* A word
* A phrase
* A meaning
* A description

The user can provide their own gesture and teach the system what that gesture represents.

### Static personalized signs

A user can capture a comfortable hand position and associate it with a meaning.

```text
Hand Pose
   ↓
Landmark Extraction
   ↓
Normalization
   ↓
Meaning
   ↓
Saved Custom Sign
```

### Dynamic personalized signs

Some gestures are defined by movement rather than a single pose.

VoiceSignAI can therefore work with motion sequences.

```text
Movement
   ↓
Multiple Camera Frames
   ↓
Landmark Sequence
   ↓
Motion Analysis
   ↓
Meaning
   ↓
Saved Custom Sign
```

Dynamic sequences are represented using a standardized sequence length of:

```text
30 frames
```

---

# ❤️ Accessibility Through Personalization

Teach My Sign changes the philosophy of gesture recognition.

Instead of:

> **"Perform the sign exactly the way the system expects."**

VoiceSignAI moves toward:

> **"Show the system how you naturally perform it."**

This can be especially valuable for users whose finger or hand mobility differs from the assumptions of a fixed vocabulary.

The personalized recognition pipeline becomes:

```text
Different User
      ↓
Different Natural Movement
      ↓
Personalized Gesture
      ↓
User-Defined Meaning
      ↓
Recognition
      ↓
Communication
```

### Our accessibility principle

> **The technology should adapt to people — not force people to adapt to the technology.**

---

# 🚀 What Makes VoiceSignAI Different?

VoiceSignAI combines multiple technologies into one accessibility-focused communication system.

## 🧠 Hybrid Recognition

The recognition architecture supports:

* Static gesture recognition
* Dynamic gesture recognition
* Motion analysis
* Landmark-based processing
* Custom user signs
* Fallback recognition

---

## ♿ Adaptive Personalization

Users can teach the system their own gestures and meanings.

This allows VoiceSignAI to move beyond a rigid one-size-fits-all gesture vocabulary.

---

## 🌍 Multiple Sign-Language Modes

The project includes architecture for:

* 🇺🇸 ASL — American Sign Language
* 🇮🇳 ISL — Indian Sign Language
* 🇬🇧 BSL — British Sign Language
* ✍️ CUSTOM — Personalized signs

---

## ✨ AI-Assisted Language Refinement

Recognized signs can be converted from fragmented words into more natural sentences using Generative AI.

---

## 🔊 Voice Communication

The final sentence can be converted into spoken audio.

---

## 🚨 Help & Emergency Awareness

The recognition backend contains a dedicated mechanism for detecting persistent help/emergency-related signals.

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
                    │    Recognition Engine       │
                    │                              │
                    │ • Static Recognition        │
                    │ • Dynamic Recognition       │
                    │ • Motion Analysis           │
                    │ • Custom Signs              │
                    │ • Fallback Recognition      │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │ Confidence & Stability Layer │
                    │                              │
                    │ • Confidence Threshold       │
                    │ • Stability Window           │
                    │ • Debouncing                 │
                    │ • Duplicate Suppression      │
                    │ • Cooldown                   │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                         ┌─────────────────────┐
                         │  Sentence Builder   │
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
                         └──────────┬──────────┘
                                    ▼
                         ┌─────────────────────┐
                         │ Spoken Communication│
                         └─────────────────────┘
```

---

# 👁️ Computer Vision Pipeline

The first stage of VoiceSignAI is visual perception.

The system uses **MediaPipe Hand Landmarker** to detect hands and extract their 3D landmarks.

Each hand contains:

```text
21 landmarks
×
3 coordinates
=
63 features
```

With two hands:

```text
63 × 2
=
126 features per frame
```

These landmark representations are then passed through normalization and recognition components.

---

# 🧠 Static & Dynamic Gesture Recognition

Not every sign can be represented by a single hand pose.

Some signs are primarily based on **hand configuration**, while others depend on **movement over time**.

VoiceSignAI therefore supports both approaches.

## Static Recognition

Static recognition can analyze:

* Finger positions
* Finger extension
* Hand shape
* Wrist position
* Relative landmark distances
* Spatial relationships

---

## Dynamic Recognition

Dynamic recognition uses sequences of frames.

The system uses:

```text
30 frames
×
126 features per frame
=
3,780 features per sequence
```

Motion analysis can consider:

* Motion energy
* Dominant hand
* Wrist trajectory
* Path length
* Net displacement
* Direction changes
* Circularity
* Two-hand activity
* Relative wrist movement

This allows movement-based gestures to be represented as temporal sequences instead of isolated images.

---

# 📐 Landmark Normalization

Raw camera coordinates can vary because of:

* Camera position
* Hand position
* Distance from the camera
* Hand scale
* User positioning

VoiceSignAI therefore includes landmark processing that can:

* Validate landmark dimensions
* Handle invalid values
* Detect NaN/Inf values
* Center coordinates around the wrist
* Normalize spatial representations
* Normalize temporal sequences

This creates a more consistent representation for recognition.

---

# 🛡️ Recognition Stability

Real-time camera predictions can be noisy.

If every frame were directly converted into a word, one gesture could become:

```text
hello hello hello hello hello
```

VoiceSignAI introduces a stability layer.

```text
Raw Prediction
      ↓
Confidence Check
      ↓
Stability Window
      ↓
Debouncing
      ↓
Cooldown
      ↓
Duplicate Suppression
      ↓
Accepted Gesture
```

This prevents repeated frame-level predictions from flooding the sentence.

---

# 📝 Sentence Construction

Recognized gestures are accumulated into a transcript.

For example:

```text
hello
   ↓
hello i
   ↓
hello i need
   ↓
hello i need water
```

The raw recognized sentence remains separate from the AI-enhanced sentence.

This creates a transparent pipeline:

```text
Gesture
  ↓
Recognized Word
  ↓
Raw Sentence
  ↓
AI Refinement
  ↓
Final Sentence
```

---

# ✨ Generative AI

Generative AI is used as a **language assistance layer**, not as a replacement for gesture recognition.

The recognition system first determines what signs were detected.

The AI then helps improve the language.

For example:

```text
Recognized:

hello i need water
```

can become:

```text
Hello, I need water.
```

The AI layer can assist with:

* Grammar
* Punctuation
* Sentence structure
* Natural phrasing
* Sentence polishing

---

# 🤖 OpenAI Integration

The Streamlit application includes an OpenAI-powered sentence correction service.

The AI call is triggered at the **sentence level**, rather than being unnecessarily called for every camera frame.

This helps reduce unnecessary API calls and keeps the recognition pipeline independent from the language model.

If AI refinement is unavailable, the raw recognized sentence can remain available.

---

# 🧠 Google Gemini Integration

The backend also includes a Gemini service layer.

Gemini functionality can be used for:

* Sentence polishing
* Sign descriptions
* Assistive language processing
* Landmark/trajectory-assisted interpretation

The backend exposes dedicated Gemini API routes.

---

# 🔊 Text-to-Speech

Once a sentence has been constructed, VoiceSignAI can convert it into spoken output.

The project includes a TTS service using **gTTS**.

```text
Recognized Signs
      ↓
Raw Sentence
      ↓
AI Refinement
      ↓
Final Sentence
      ↓
gTTS
      ↓
MP3 Audio
      ↓
Playback
```

The Streamlit interface can also make use of browser speech capabilities for immediate voice feedback.

---

# 🚨 Help & Emergency Detection

VoiceSignAI contains a dedicated `HelpDetector`.

The detector monitors recognition results for help/emergency-related concepts such as:

```text
help
emergency
assistance
need help
danger
sos
```

The system does not immediately trigger an event from a single prediction.

Instead, the signal is required to persist for a configured duration.

The current backend configuration uses:

```text
5 seconds
```

as the persistence window.

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

This architecture helps distinguish persistent intentional signals from momentary recognition noise.

---

# 🌍 Multilingual Data Architecture

VoiceSignAI contains separate data and ingestion structures for:

```text
ASL
ISL
BSL
CUSTOM
```

The ingestion layer supports multiple dataset formats and provides adapters for different language sources.

The system also includes synthetic data generation capabilities for development and testing.

---

# 🗃️ Custom Sign Data

Custom signs are persisted using **SQLite**.

Stored information can include:

* Sign label
* Sign description
* User association
* Static samples
* Dynamic samples
* Landmark representations
* Motion metadata

This allows personalized signs to remain available between sessions.

---

# 🖥️ Application Interface

The primary interface is built using **Streamlit**.

## 🔍 Lens Mode

The main interaction experience provides functionality for:

* Language selection
* Camera interaction
* Hand landmark visualization
* Gesture recognition
* Sentence construction
* AI sentence improvement
* Voice output
* Undo
* Clear
* Reset
* Demo functionality

---

## ✍️ Custom Sign Studio

The Custom Sign Studio provides:

* Static sign enrollment
* Dynamic sign recording
* Dynamic video upload
* Custom sign storage
* Sign management
* Sample management
* Voice testing

Most importantly, it gives users the ability to create a vocabulary that matches **their own natural way of signing**.

---

# 🎬 Demo Mode

VoiceSignAI contains a demo mechanism that can be used when a live ML model or camera environment is not ideal.

This allows the complete pipeline to be demonstrated:

```text
Demo Gesture
     ↓
Recognition
     ↓
Sentence
     ↓
AI Refinement
     ↓
Voice
```

This is useful for hackathon demonstrations, testing, and development.

---

# ⚡ Backend Architecture

VoiceSignAI also includes a modular **FastAPI backend**.

The backend provides REST and WebSocket infrastructure for:

* Real-time recognition
* Transcript management
* Custom signs
* Dataset management
* Gemini services
* Help detection
* Recognition events

This makes the project extensible beyond the current Streamlit interface.

---

# 🔌 API Overview

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

# 🗂️ Project Structure

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

# 🧰 Technology Stack

| Layer                      | Technology                |
| -------------------------- | ------------------------- |
| 🎨 UI                      | Streamlit                 |
| 👁️ Computer Vision        | MediaPipe, OpenCV         |
| ✋ Hand Tracking            | MediaPipe Hand Landmarker |
| 🔢 Numerical Processing    | NumPy                     |
| 🧠 Machine Learning        | scikit-learn, joblib      |
| 🤖 Generative AI           | OpenAI, Google Gemini     |
| 🔊 Text-to-Speech          | gTTS                      |
| ⚡ Backend                  | FastAPI                   |
| 🔄 Real-Time Communication | WebSockets                |
| 🗃️ Database               | SQLite                    |
| 📊 Dataset Processing      | JSONL, CSV, JSON          |
| 🧪 Testing                 | pytest                    |
| ⚙️ Configuration           | python-dotenv             |

---

# 📊 Technical Representation

VoiceSignAI uses hand landmarks as the foundation of its gesture representation.

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

For a dynamic sequence:

```text
126 features / frame
×
30 frames
=
3,780 features / sequence
```

These representations support landmark processing, motion analysis, custom-sign storage, and model-based recognition.

---

# 🔄 Complete End-to-End Pipeline

```text
                 🤟 USER SIGN
                       │
                       ▼
              🎥 CAMERA INPUT
                       │
                       ▼
             👁️ HAND DETECTION
                       │
                       ▼
              📍 LANDMARKS
                       │
                       ▼
              📐 NORMALIZATION
                       │
                       ▼
             🧠 RECOGNITION
              /       |       \
             /        |        \
        STATIC     DYNAMIC    CUSTOM
             \        |        /
              \       |       /
               ▼      ▼      ▼
             🛡️ STABILITY FILTER
                       │
                       ▼
                 📝 SENTENCE
                       │
                       ▼
                ✨ AI REFINEMENT
                       │
                       ▼
                  🔊 SPEECH
                       │
                       ▼
                🗣️ COMMUNICATION
```

---

# 🧠 Recognition + Personalization

The most important relationship in the system is:

```text
General Vocabulary
        +
Personal Vocabulary
        ↓
VoiceSignAI Recognition
```

The system can therefore combine predefined recognition infrastructure with signs learned specifically from the user.

This is particularly important for accessibility because personalization can help accommodate differences in how individual users naturally perform gestures.

---

# 🔐 Configuration

VoiceSignAI uses environment variables for optional AI services and configurable recognition parameters.

Create your environment configuration from:

```text
.env.example
```

OpenAI configuration:

```env
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o-mini
OPENAI_TIMEOUT=10.0
```

Gemini configuration:

```env
GEMINI_API_KEY=your_key_here
GEMINI_MODEL_NAME=gemini-2.5-flash
```

Optional recognition configuration:

```env
CONFIDENCE_THRESHOLD=0.65
```

API credentials should never be committed to the repository.

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

Create a `.env` file using `.env.example` as the reference.

Configure the AI services you want to use.

VoiceSignAI is designed so that optional AI services do not have to make the entire application unusable.

---

## 5. Start the Streamlit application

```bash
streamlit run app.py
```

The VoiceSignAI interface will then be available through the local Streamlit URL.

---

# 🌐 Starting the FastAPI Backend

Run:

```bash
python run_server.py
```

Or:

```bash
python run_server.py --host 127.0.0.1 --port 8000 --reload
```

FastAPI's interactive API documentation will be available at:

```text
http://127.0.0.1:8000/docs
```

---

# 🧪 Testing

VoiceSignAI contains an automated test suite covering major components of the system.

Testing includes areas such as:

* API endpoints
* Custom signs
* Dataset management
* Gemini integration
* Gesture processing
* Landmark processing
* Recognition
* Sentence construction
* Application state
* Text-to-speech

Run:

```bash
pytest -v
```

---

# 🎬 Hackathon Demo Flow

For a strong demonstration of VoiceSignAI, the complete experience can be presented as follows.

### 1️⃣ Start the application

Launch the Streamlit interface.

### 2️⃣ Select a sign-language mode

Choose the required language mode.

### 3️⃣ Demonstrate recognition

Show a gesture to the camera.

The system extracts landmarks and processes the gesture.

### 4️⃣ Build a sentence

Perform multiple signs and demonstrate how the recognized words accumulate into a sentence.

### 5️⃣ Improve the sentence

Use the AI refinement functionality to transform fragmented recognition output into natural language.

### 6️⃣ Speak

Use the speech functionality to convert the final sentence into voice.

### 7️⃣ Demonstrate the accessibility feature

Open **Teach My Sign**.

Perform a gesture in a way that is comfortable and natural for the user.

Give that gesture a meaning such as:

```text
Water
Help
Food
Hello
Thank You
```

Save it.

Then perform the personalized gesture again and demonstrate its recognition.

### 8️⃣ Demonstrate dynamic personalization

Show how a movement-based custom sign can be recorded and stored.

### 9️⃣ Demonstrate help detection

Show the architecture for persistent HELP/EMERGENCY recognition.

### 🔟 Demonstrate the backend

Open the FastAPI documentation and show the modular REST/WebSocket API architecture.

---

# 💭 A Simple Example

Imagine a user wants to communicate:

> "Hello, I need water."

The interaction can look like:

```text
🤟 Hello
      ↓
🧠 "hello"

🤟 I
      ↓
🧠 "i"

🤟 Need
      ↓
🧠 "need"

🤟 Water
      ↓
🧠 "water"
```

The system builds:

```text
hello i need water
```

Then the AI layer can refine it:

```text
Hello, I need water.
```

Finally:

```text
🔊 "Hello, I need water."
```

Now imagine that the user performs the "water" gesture differently because of their individual hand or finger mobility.

With **Teach My Sign**, they can teach VoiceSignAI:

```text
My gesture → "water"
```

The system can then use that personalized gesture as part of their communication vocabulary.

---

# 🌟 Why This Matters

VoiceSignAI is not trying to create a system where every person must communicate in exactly the same physical way.

Instead, it explores a more adaptive model:

```text
             Traditional Systems
                    │
                    ▼
          Fixed Gesture Vocabulary
                    │
                    ▼
              User adapts
```

versus:

```text
                VoiceSignAI
                    │
                    ▼
          User's Natural Gesture
                    │
                    ▼
             Teach My Sign
                    │
                    ▼
            User-defined Meaning
                    │
                    ▼
          Personalized Vocabulary
```

This is where the project moves beyond simple sign recognition and toward **personalized assistive communication**.

---

# 🏗️ Development Philosophy

VoiceSignAI is organized into independent modules so that different parts of the system can evolve independently.

```text
Computer Vision
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
      ├── Stability
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

This modularity makes it possible to improve recognition models, add new languages, expand personalization, or replace individual AI services without redesigning the entire system.

---

# ⚡ Graceful Degradation

VoiceSignAI is designed to remain useful even when optional components are unavailable.

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

Similarly, the architecture includes fallback recognition and demo functionality for environments where trained model assets or live recognition are not available.

The goal is to keep the overall application flow functional instead of allowing one optional dependency to break the entire experience.

---

# 🔮 Future Scope

VoiceSignAI provides a foundation for continued development.

Future improvements can include:

* Larger real-world sign-language datasets
* Expanded ISL vocabulary
* Expanded ASL vocabulary
* Expanded BSL vocabulary
* Improved signer-independent recognition
* More advanced temporal gesture models
* Stronger two-hand recognition
* More personalized recognition capabilities
* User-specific adaptive models
* Offline AI capabilities
* Offline speech synthesis
* Mobile deployment
* Edge-device inference
* User profiles
* Real-time multi-user communication
* Recognition accuracy benchmarking
* Latency benchmarking
* Expanded emergency communication features
* Additional accessibility-focused features

---

# ❤️ Our Vision

VoiceSignAI started with a simple question:

> **What if technology could learn the way a person communicates instead of forcing the person to change the way they communicate?**

That question shaped the **Teach My Sign** feature and the rest of the system.

A sign can represent:

* A greeting
* A request
* A need
* A thought
* A conversation
* An emergency

And every person may express those ideas differently.

VoiceSignAI brings together:

**👁️ Computer Vision**

to see the hands,

**🧠 Gesture Recognition**

to understand movement,

**♿ Personalization**

to adapt to individual users,

**✨ Generative AI**

to improve language,

and

**🔊 Speech**

to give the message a voice.

---

# 🤟 Sign → Understand → Adapt → Refine → Speak

```text
             🤟 SIGN
                │
                ▼
          👁️ SEE THE HAND
                │
                ▼
        🧠 UNDERSTAND GESTURE
                │
                ▼
          ♿ ADAPT TO USER
                │
                ▼
            📝 BUILD
                │
                ▼
            ✨ REFINE
                │
                ▼
             🔊 SPEAK
                │
                ▼
          ❤️ COMMUNICATE
```

---

<p align="center">

# 🤟 VoiceSignAI

### **Let every sign have a voice.**

### **And let every person sign in their own way.**

**Built by Sloppy Minds**

**Hackverse: Into the Web**

</p>
