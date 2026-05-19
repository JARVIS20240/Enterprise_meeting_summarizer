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
  - **Note on Custom Installations**: If you installed Ollama in a custom location, ensure its executable is added to your System PATH so you can use it from the command line.
  - Once installed, open a terminal and run: `ollama pull gemma4:e2b`
- **wkhtmltopdf**: Required for PDF generation. [Download here](https://wkhtmltopdf.org/downloads.html).

---

## 3. Step-by-Step Setup Guide (For Windows)
Follow these steps carefully to set up the project on your local machine.

### Step 1: Clone the Repository
Open your terminal (CMD or PowerShell) and run:
```bash
git clone <your-repo-url>
cd Enterprise_Meeting_Summarizer
```

### Step 2: Create & Activate Virtual Environment
```bash
python -m venv Enterprise
Enterprise\Scripts\activate
```

### Step 3: Install Core Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Install PyTorch with CUDA 12.1 Support
This is critical for GPU acceleration. Do not skip this step:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Step 5: Configure Environment Variables
Create a file named `.env` in the root directory (`Enterprise_Meeting_Summarizer`) with the following content:
```env
HUGGINGFACE_TOKEN=your_hf_token_here
NGROK_AUTH_TOKEN=your_ngrok_token_here
```
> **Note**: Get your HF token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). You must accept the user conditions for `pyannote/speaker-diarization-3.1` on HuggingFace for the pipeline to work.

---

## 4. ⚠️ CRITICAL CONFIGURATION: Custom Hardcoded Paths (MUST READ) ⚠️
This application was originally developed using custom drive paths (`M:\` and `D:\`). **You MUST update the paths in the following files** to match your local computer's directory structure, or the app will crash.

### 📁 1. Model Download Paths (`M:\` Drive)
The app stores heavy AI models in a specific directory to avoid bloating the C: drive. Update the paths below to your preferred location (e.g., `C:\AI_Models\Voice_Separator` and `C:\AI_Models\Whisper`).

**Update `VOICE_MODEL_PATH` and `WHISPER_MODEL_PATH` in these files:**
- `core/merger_engine.py` (Lines 21, 22)
- `core/biometric_worker.py` (Line 53)
- `core/pyannote_worker.py` (Line 53)
- `core/voice_biometrics.py` (Line 21)
- `scripts/run_diarization.py` (Line 20)
- `scripts/run_transcription.py` (Line 13)

*Change `r"M:\Models\..."` to a valid path on your PC where you want models to be stored.*

### 📄 2. wkhtmltopdf Executable Path (`D:\` Drive)
The PDF engine needs to know where the `wkhtmltopdf` converter is installed.
- **File to Edit**: `app/Meeting_summary.py` (near line 414)
- **Variable to Change**: `path_wkhtmltopdf`
- **Action**: Update `r'D:\Softwares\wkhtmltopdf\bin\wkhtmltopdf.exe'` to the exact location where you installed `wkhtmltopdf.exe` (e.g., `r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe'`).

---

## 5. How to Run the Application

### Option A: Standard Local Run (Streamlit)
Start the main application interface locally:
```bash
streamlit run app\Meeting_summary.py --server.fileWatcherType none
```

### Option B: Remote Access (Ngrok Tunnel)
Launch Streamlit in the background and generate a public URL to share the interface securely:
```bash
python utils/ngrok_launcher.py
```

---

## 6. File Structure Reference
```text
Enterprise_Meeting_Summarizer/
├── .env                          # Configuration file for environment variables and API tokens.
├── .gitignore                    # Specifies intentionally untracked files to ignore.
├── README.md                     # Project overview and documentation for setup and usage.
├── requirements.txt              # List of Python dependencies required for the project environment.
├── app/                          # Main user interface components and application logic.
│   ├── Meeting_summary.py        # Main Streamlit UI for audio processing and summary generation.
│   └── pages/                    # Multi-page Streamlit navigation for specialized tasks.
├── core/                         # Core AI processing engines and backend logic.
│   ├── biometric_worker.py       # Isolated process for extracting voice embeddings securely.
│   ├── merger_engine.py          # Orchestration engine for diarization, biometrics, and transcription.
│   ├── pyannote_worker.py        # Sandboxed process for running speaker diarization patterns.
│   └── voice_biometrics.py       # Core logic for extracting 512-D voice fingerprints.
├── data/                         # Local storage for databases, audio artifacts, and model data.
├── scripts/                      # Standalone command-line utilities for manual execution.
└── utils/                        # Shared utility functions and helper modules.
```
