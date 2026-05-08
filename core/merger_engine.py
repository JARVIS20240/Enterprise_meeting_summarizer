import os
import gc
from pathlib import Path
from dotenv import load_dotenv

# This strictly prevents the silent OpenMP C++ crash on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ==========================================
# 1. DYNAMIC PATH ANCHORING & MODEL CACHE
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

load_dotenv(BASE_DIR / ".env")

VOICE_MODEL_PATH = r"M:\Models\Voice_Separator"
WHISPER_MODEL_PATH = r"M:\Models\Whisper"

os.environ["HF_HOME"] = VOICE_MODEL_PATH
os.environ["TORCH_HOME"] = VOICE_MODEL_PATH
os.environ["PYANNOTE_CACHE"] = VOICE_MODEL_PATH

hf_token = os.getenv("HUGGINGFACE_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
else:
    print("❌ ERROR: HUGGINGFACE_TOKEN not found in .env file!")

import logging
import threading
import subprocess
import tempfile

# ══════════════════════════════════════════════════════════
# 2. AUDIO & TRANSCRIPTION CORE
# ══════════════════════════════════════════════════════════
SUPPORTED_VIDEO  = [".mp4", ".mov", ".avi", ".mkv"]
SUPPORTED_AUDIO  = [".mp3", ".wav", ".m4a", ".ogg", ".flac"]

def clear_vram():
    """Forcefully clears GPU memory."""
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        
def transcribe_audio(wav_path: str, live_status_container=None):
    import torch
    import sys
    import json
    
    has_cuda = torch.cuda.is_available()
    device_type = "cuda" if has_cuda else "cpu"
    
    if live_status_container:
        live_status_container.markdown("**Analyzing speaker patterns... (Phase A)** *(Running Sandboxed Process)*")
        
    out_json = wav_path + ".diarization.json"
    worker_script = str(BASE_DIR / "core" / "pyannote_worker.py")
    
    python_exe = sys.executable
    cmd = [python_exe, worker_script, wav_path, out_json, hf_token]
    
    logging.info("Spawning isolated Pyannote worker process...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if not os.path.exists(out_json):
        raise RuntimeError(f"Pyannote worker crashed fatally (C++ error or OOM):\n{result.stderr}")
        
    with open(out_json, "r") as f:
        data = json.load(f)
        
    if isinstance(data, dict) and "error" in data:
        raise RuntimeError(f"Pyannote Error: {data['error']}\n{data.get('traceback','')}")
        
    diarization_segments = data
    logging.info("Diarization complete. VRAM is 100% cleanly flushed by OS.")

    # --- PHASE B: TRANSCRIPTION (WHAT & WHEN) ---
    logging.info("Starting Phase B: Faster-Whisper Transcription...")
    if live_status_container:
        live_status_container.markdown("**Transcribing audio... (Phase B)**")

    compute_type = "float16" if has_cuda else "int8"
    
    logging.info(f"Loading Whisper into {device_type.upper()} with {compute_type} precision...")
    from faster_whisper import WhisperModel
    
    model = WhisperModel("base", device=device_type, compute_type=compute_type, cpu_threads=8, download_root=WHISPER_MODEL_PATH)
    
    logging.info("Executing ultra-fast transcription sequence...")
    segments, info = model.transcribe(
        wav_path,
        beam_size=1,            
        vad_filter=True, 
        vad_parameters=dict(
            min_silence_duration_ms=500
        ),
        condition_on_previous_text=False, 
        without_timestamps=False,
        task="translate"
    )

    segments_list = []
    for segment in segments:
        segments_list.append(segment)
        if live_status_container:
            live_status_container.markdown(f"**Transcribing... ({info.language})** *(Keeping connection alive)*\n\n> {segment.text.strip()}")

    duration_seconds = info.duration
    
    logging.info("Transcription complete. Flushing VRAM for Alignment...")
    del model
    clear_vram()
    
    # --- PHASE C: ALIGNMENT ---
    logging.info("Starting Phase C: Alignment...")
    if live_status_container:
        live_status_container.markdown("**Aligning speakers with transcript... (Phase C)**")
        
    final_transcript_lines = []
    for segment in segments_list:
        segment_center = segment.start + (segment.end - segment.start) / 2
        assigned_speaker = "UNKNOWN_SPEAKER"
        
        for track in diarization_segments:
            if track["start"] <= segment_center <= track["end"]:
                assigned_speaker = track["speaker"]
                break
        
        text = segment.text.strip()
        line = f"{assigned_speaker}: {text}"
        final_transcript_lines.append(line)
        
    transcript = "\n".join(final_transcript_lines).strip()
    
    return transcript, duration_seconds, info.language