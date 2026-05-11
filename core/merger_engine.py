import os
import gc
from pathlib import Path
from dotenv import load_dotenv

# This strictly prevents the silent OpenMP C++ crash on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import warnings
# Silence the deprecated torchaudio backend warning from pyannote.audio
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.core.io")

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

    # --- PHASE A.5: TIERED CONFIDENCE MERGE ---
    if live_status_container:
        live_status_container.markdown("**Comparing Voice Biometrics with Database... (Phase A.5)**")
        
    speaker_map = {}
    
    try:
        import chromadb
        from core.voice_biometrics import VoiceEmbeddingEngine
        
        DATA_DIR = BASE_DIR / "data"
        chroma_client = chromadb.PersistentClient(path=str(DATA_DIR / "chroma_db"))
        collection = chroma_client.get_or_create_collection(
            name="employee_voices_v2",
            metadata={"hnsw:space": "cosine"}
        )
        
        if collection.count() > 0:
            logging.info(f"Biometrics DB found {collection.count()} enrolled profiles. Cross-referencing...")
            
            # Step 1: Collect all segments to extract
            extraction_tasks = []
            task_to_speaker = {}
            
            speaker_segments_map = {}
            for track in diarization_segments:
                spk = track["speaker"]
                duration = track["end"] - track["start"]
                if spk not in speaker_segments_map:
                    speaker_segments_map[spk] = []
                speaker_segments_map[spk].append({"start": track["start"], "end": track["end"], "duration": duration})
            
            for spk, segments in speaker_segments_map.items():
                segments.sort(key=lambda x: x["duration"], reverse=True)
                top_segments = segments[:1]
                
                for i, seg in enumerate(top_segments):
                    temp_wav = str(DATA_DIR / f"temp_golden_{spk}_{i}.wav")
                    task_id = f"{spk}_{i}"
                    
                    # Extract audio segment
                    subprocess.run(["ffmpeg", "-y", "-ss", str(seg["start"]), "-to", str(seg["end"]), "-i", wav_path, "-ac", "1", "-ar", "16000", temp_wav], 
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    
                    if os.path.exists(temp_wav):
                        extraction_tasks.append({"id": task_id, "path": temp_wav})
                        task_to_speaker[task_id] = spk

            # Step 2: Spawn isolated process for bulk embedding extraction
            if extraction_tasks:
                logging.info(f"Spawning biometric worker for {len(extraction_tasks)} segments...")
                task_json = wav_path + ".bio_tasks.json"
                result_json = wav_path + ".bio_results.json"
                
                with open(task_json, 'w') as f:
                    json.dump(extraction_tasks, f)
                
                worker_script = str(BASE_DIR / "core" / "biometric_worker.py")
                python_exe = sys.executable
                cmd = [python_exe, worker_script, task_json, result_json, hf_token]
                
                worker_result = subprocess.run(cmd, capture_output=True, text=True)
                
                if os.path.exists(result_json):
                    with open(result_json, 'r') as f:
                        bio_data = json.load(f)
                    
                    if "results" in bio_data:
                        embeddings = bio_data["results"]
                        
                        # Group results by speaker
                        speaker_best_match = {}
                        
                        for task_id, vec in embeddings.items():
                            if vec is None: continue
                            
                            spk = task_to_speaker[task_id]
                            
                            # Query ChromaDB
                            results = collection.query(query_embeddings=[vec], n_results=1)
                            
                            if results['distances'] and len(results['distances'][0]) > 0:
                                dist = results['distances'][0][0]
                                meta = results['metadatas'][0][0]
                                
                                if spk not in speaker_best_match or dist < speaker_best_match[spk]["dist"]:
                                    speaker_best_match[spk] = {
                                        "dist": dist,
                                        "name": meta.get("name", "Unknown"),
                                        "role": meta.get("role", "")
                                    }
                        
                        # Step 3: Apply tiered matching logic
                        for spk, match in speaker_best_match.items():
                            dist = match["dist"]
                            name = match["name"]
                            role = match["role"]
                            
                            if dist < 0.3:
                                speaker_map[spk] = f"{name} ({role})" if role else name
                                logging.info(f"High Match: {spk} -> {name} (Dist: {dist:.4f})")
                            elif dist < 0.45:
                                speaker_map[spk] = f"Likely {name} (Unverified)"
                                logging.info(f"Med Match: {spk} -> {name} (Dist: {dist:.4f})")
                            else:
                                speaker_map[spk] = f"Guest ({spk})"
                else:
                    logging.error(f"Biometric worker failed:\n{worker_result.stderr}")
                
                # Cleanup temp files
                for task in extraction_tasks:
                    if os.path.exists(task["path"]): os.remove(task["path"])
                if os.path.exists(task_json): os.remove(task_json)
                if os.path.exists(result_json): os.remove(result_json)

            # Fill in remaining speakers as guests
            for track in diarization_segments:
                spk = track["speaker"]
                if spk not in speaker_map:
                    speaker_map[spk] = f"Guest ({spk})"

    except Exception as e:
        logging.error(f"Tiered Confidence Merge failed: {e}")

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
                assigned_speaker = speaker_map.get(track["speaker"], f'Guest ({track["speaker"]})')
                break
        
        text = segment.text.strip()
        line = f"{assigned_speaker}: {text}"
        final_transcript_lines.append(line)
        
    transcript = "\n".join(final_transcript_lines).strip()
    
    return transcript, duration_seconds, info.language