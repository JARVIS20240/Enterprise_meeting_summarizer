# Enterprise Meeting Summarizer 🚀

A high-performance, fully local, AI-powered pipeline for transcribing, identifying speakers, and generating executive meeting reports. Designed for maximum privacy, this application runs entirely on your local hardware using cutting-edge open-source models.

## 1. Project Overview & Features
The **Enterprise Meeting Summarizer** is an all-in-one solution for corporate documentation. It processes audio/video files through a sophisticated multi-stage pipeline:
- **GPU-Accelerated Transcription**: Uses `faster-whisper` on CUDA for ultra-fast, high-accuracy text extraction.
- **Pyannote Speaker Diarization**: Automatically detects different speakers and calculates "who spoke when."
- **ChromaDB Voice Biometrics**: Cross-references speaker fingerprints against a local vector database to identify employees by name.
- **Local Ollama LLM Integration**: Uses a "Map-Reduce" engine with `gemma4:e2b` to summarize long transcripts without context window limits.
- **Premium PDF Generation**: Produces enterprise-grade executive reports with custom HTML/CSS styling.
- **Remote Access**: Built-in support for Ngrok tunneling to share the interface securely.

---

## 2. System Prerequisites (Strict)
To run this project, your machine **must** meet the following requirements:

- **Python**: Version **3.12** is strictly required.
- **NVIDIA GPU**: Minimum **4GB VRAM** (e.g., RTX 3050). CUDA Toolkit must be installed.
- **FFmpeg**: Must be installed and added to your **System PATH**. [Download here](https://ffmpeg.org/download.html).
- **Ollama**: Must be installed and running locally. [Download here](https://ollama.com/).
  - Once installed, run: `ollama pull gemma4:e2b`
- **wkhtmltopdf**: Required for PDF generation. [Download here](https://wkhtmltopdf.org/downloads.html).

---

## 3. Step-by-Step Setup Guide (For Windows)
Open your terminal (CMD or PowerShell) in the project root and run these commands:

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd Enterprise_Meeting_Summarizer
```

### 2. Create & Activate Virtual Environment
```bash
python -m venv Enterprise
Enterprise\Scripts\activate
```

### 3. Install Core Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install PyTorch with CUDA 12.1 Support
This is critical for GPU acceleration. Do not skip this step:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## 4. ⚠️ CRITICAL CONFIGURATION: Custom Hardcoded Paths (MUST READ) ⚠️
This application is optimized for a specific hardware setup. **You MUST update the following paths in the code** to match your own computer, or the app will crash immediately.

### 📁 Model Download Paths
The app stores heavy AI models in a specific directory to avoid bloating your C: drive.
- **Files to Edit**: `core/merger_engine.py`, `core/voice_biometrics.py`, `core/pyannote_worker.py`, and `core/biometric_worker.py`.
- **Variable to Change**: `VOICE_MODEL_PATH` and `WHISPER_MODEL_PATH`.
- **Action**: Change `r"M:\Models\..."` to a path that exists on your PC (e.g., `r"C:\AI_Models\..."`).

### 📄 wkhtmltopdf Executable Path
The PDF engine needs to know where the converter is installed.
- **File to Edit**: `app/Meeting_summary.py` (near line 412).
- **Variable to Change**: `path_wkhtmltopdf`.
- **Action**: Update `r'D:\Softwares\wkhtmltopdf\bin\wkhtmltopdf.exe'` to your actual installation path.

### 🔑 HuggingFace & Ngrok Tokens
You must create a `.env` file in the root directory with the following content:
```env
HUGGINGFACE_TOKEN=your_token_here
NGROK_AUTH_TOKEN=your_token_here
```
> **Note**: Get your HF token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (ensure you have accepted the terms for `pyannote/speaker-diarization-3.1`).

---

## 5. How to Run

### Option A: Standard Local Run (Streamlit)
```bash
streamlit run app/Meeting_summary.py
```

### Option B: Remote Access (Ngrok Tunnel)
This will launch Streamlit in the background and provide a public URL.
```bash
python utils/ngrok_launcher.py
```

---

## 6. File Structure
Enterprise_Meeting_Summarizer/
├── .env                          # Configuration file for environment variables and API tokens.
├── .gitignore                    # Specifies intentionally untracked files to ignore.
├── README.md                     # Project overview and documentation for setup and usage.
├── requirements.txt              # List of Python dependencies required for the project environment.
├── system_debug.log              # Centralized log file for tracking system errors and events.
├── app/                          # Main user interface components and application logic.
│   ├── Meeting_summary.py        # Main Streamlit UI for audio processing and summary generation.
│   └── pages/                    # Multi-page Streamlit navigation for specialized tasks.
│       ├── 1_Voice_Enrollment.py # Interface for enrolling new speaker voices into the database.
│       └── 2_Database_Manager.py # Management tool for viewing and deleting speaker profiles.
├── core/                         # Core AI processing engines and backend logic.
│   ├── biometric_worker.py       # Isolated process for extracting voice embeddings securely.
│   ├── merger_engine.py          # Orchestration engine for diarization, biometrics, and transcription.
│   ├── pyannote_worker.py        # Sandboxed process for running speaker diarization patterns.
│   └── voice_biometrics.py       # Core logic for extracting 512-D voice fingerprints.
├── data/                         # Local storage for databases, audio artifacts, and model data.
│   ├── DEBUG_notes_dump.txt      # Temporary file containing raw AI-generated shorthand notes.
│   ├── file_structure.txt        # Legacy file structure mapping for reference.
│   ├── temp_golden.wav           # Temporary audio segment used for biometric verification.
│   └── test_meeting.wav          # Sample audio file for testing the transcription pipeline.
├── scripts/                      # Standalone command-line utilities for manual execution.
│   ├── run_diarization.py        # CLI script for performing speaker diarization independently.
│   └── run_transcription.py      # CLI script for transcribing audio files using Whisper.
└── utils/                        # Shared utility functions and helper modules.
    ├── logs.py                   # Centralized logging configuration for debugging.
    └── ngrok_launcher.py         # Utility for tunneling the local server to a public URL.
