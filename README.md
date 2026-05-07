# Enterprise Meeting Summarizer

A high-performance, fully local, AI-powered pipeline for transcribing, summarizing, and generating enterprise-grade PDF executive reports from meeting audio files. This pipeline is optimized to run locally on consumer GPUs (e.g., RTX 3050) without relying on cloud APIs, ensuring maximum data privacy and security.

## Features
- **Extreme Speed Transcription**: Utilizes `faster-whisper` (CTranslate2) on CUDA for high-speed transcription with Voice Activity Detection (VAD) to skip silences automatically.
- **Map-Reduce Summarization**: Leverages local Ollama LLMs (e.g., `gemma:2b`) to process extremely long meetings in chunks to bypass context window limitations.
- **Executive PDF Generation**: Converts structured Markdown meeting summaries into beautifully formatted, enterprise-grade PDFs using HTML/CSS templating and `wkhtmltopdf`.
- **Advanced Memory Management**: Explicit PyTorch VRAM flushing sequences to safely transition GPU memory between transcription models and LLMs, preventing Out-Of-Memory (OOM) crashes.
- **Live GPU Telemetry**: Real-time VRAM monitoring integrated directly into the Streamlit UI.

---

## Prerequisites

Before setting up the project, ensure you have the following installed on your system:

1. **Python**: Exactly Version 3.12.10 (Tested Environment).
2. **NVIDIA GPU**: Minimum 4GB VRAM (tested successfully on RTX 3050) with CUDA Toolkit installed.
3. **FFmpeg**: Required for audio extraction. Must be installed and added to your system's `PATH`.
4. **wkhtmltopdf**: Required for PDF rendering. 
   - Download the Windows installer from the official website.
   - Install it to your preferred location (the script is currently configured to look for `D:\Softwares\wkhtmltopdf\bin\wkhtmltopdf.exe`). If installed elsewhere, update the path in `Meeting_summary.py`.
5. **Ollama**: Required to run the local LLMs. Install Ollama and verify it is running in the background.

---

## Installation Guide

Follow these steps exactly to configure the local environment:

### 1. Clone the Repository
```bash
git clone <your-repository-url>
cd Enterprise_Meeting_Summarizer
```

### 2. Create the Virtual Environment
We isolate dependencies inside a virtual environment.
```bash
python -m venv Enterprise
```

### 3. Activate the Environment
**Windows (Command Prompt):**
```cmd
Enterprise\Scripts\activate.bat
```
**Windows (PowerShell):**
```powershell
.\Enterprise\Scripts\Activate.ps1
```

### 4. Install Dependencies
First, install the required packages with strictly pinned versions:
```bash
pip install -r requirements.txt
```

Then, manually install PyTorch with CUDA 12.1 support using the exact index URL:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 5. Download Local LLM Models
Pull the required model into your local Ollama instance. The script defaults to `gemma:2b`.
```bash
ollama pull gemma:2b
```

---

## Running the Application

1. Ensure Ollama is running in your system tray.
2. Activate your virtual environment if not already active.
3. Start the Streamlit application:
```bash
streamlit run Meeting_Summary/Meeting_summary.py
```
4. Upload an audio or video file via the web interface to begin the pipeline.

---

## File Structure

```text
Enterprise_Meeting_Summarizer/
├── .gitignore                      # Git exclusions (ignores logs, venv, temporary media)
├── requirements.txt                # Python package dependencies
├── README.md                       # Documentation
├── Meeting_Summary/
│   ├── Meeting_summary.py          # Main Streamlit App: handles UI, transcription, LLM orchestration, and PDF generation
│   ├── logs.py                     # GPU Telemetry script: parses nvidia-smi for VRAM utilization
│   └── system_debug.log            # Main application log file (auto-generated during execution)
└── Enterprise/                     # Virtual Environment (Generated during setup, excluded from git)
```

## Technical Notes

- **Modifying Prompts**: The summarization prompts are located within `Meeting_summary.py`. When making changes, ensure you retain the Markdown formatting (`## Overview`, `## Action Items`, etc.) as the PDF rendering engine relies on standard markdown syntaxes to apply CSS styling.
- **Troubleshooting OOM (Out of Memory)**: If the system crashes during the transition from Transcription to Summarization, ensure your background applications (like browsers) are closed to free up the baseline VRAM needed. The script automatically executes `torch.cuda.empty_cache()` to assist with memory hand-offs.
