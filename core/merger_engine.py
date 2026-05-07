import os
import torch
import gc 
from pathlib import Path
from pyannote.audio import Pipeline
from faster_whisper import WhisperModel
from dotenv import load_dotenv

# ==========================================
# 1. DYNAMIC PATH ANCHORING (Enterprise Grade)
# ==========================================
# This finds the absolute path to 'Enterprise_Meeting_Summarizer' 
# no matter where you run the script from.
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
AUDIO_INPUT = str(DATA_DIR / "test_meeting.wav")
TRANSCRIPT_OUTPUT = str(DATA_DIR / "final_transcript.txt")

# LOAD SECRETS FROM .ENV
load_dotenv(BASE_DIR / ".env")

# ==========================================
# 2. ARCHITECTURE LOCKS
# ==========================================
VOICE_MODEL_PATH = r"M:\Models\Voice_Separator"
WHISPER_MODEL_PATH = r"M:\Models\Whisper"

os.environ["HF_HOME"] = VOICE_MODEL_PATH
os.environ["TORCH_HOME"] = VOICE_MODEL_PATH
os.environ["PYANNOTE_CACHE"] = VOICE_MODEL_PATH

# SECURE TOKEN LOADING
hf_token = os.getenv("HUGGINGFACE_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
else:
    print("❌ ERROR: HF_TOKEN not found in .env file!")

def clear_vram():
    """Forcefully clears GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

# ==========================================
# 3. PHASE A: DIARIZATION (WHO & WHEN)
# ==========================================
print("🚀 STARTING PHASE A: DIARIZATION...")
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
if torch.cuda.is_available():
    pipeline.to(torch.device("cuda"))

print(f"🧠 Analyzing speaker patterns in: {AUDIO_INPUT}")
diarization_result = pipeline(AUDIO_INPUT)

print("🧹 Purging Diarization model from VRAM...")
del pipeline
clear_vram()

# ==========================================
# 4. PHASE B: TRANSCRIPTION (WHAT & WHEN)
# ==========================================
print("\n🚀 STARTING PHASE B: TRANSCRIPTION...")
model = WhisperModel("base", device="cuda", compute_type="float16", download_root=WHISPER_MODEL_PATH)

print("🧠 Transcribing audio text...")
segments, _ = model.transcribe(AUDIO_INPUT, beam_size=5)
segments = list(segments) 

print("🧹 Purging Whisper model from VRAM...")
del model
clear_vram()

# ==========================================
# 5. PHASE C: ALIGNMENT & OUTPUT
# ==========================================
print("\n" + "=" * 60)
print("📜 FINAL MERGED TRANSCRIPT")
print("=" * 60)

with open(TRANSCRIPT_OUTPUT, "w", encoding="utf-8") as f:
    for segment in segments:
        segment_center = segment.start + (segment.end - segment.start) / 2
        assigned_speaker = "UNKNOWN"
        
        for turn, _, speaker in diarization_result.itertracks(yield_label=True):
            if turn.start <= segment_center <= turn.end:
                assigned_speaker = speaker
                break
        
        # strip() removes leading spaces from Whisper's output
        text = segment.text.strip()
        line = f"[{segment.start:05.1f}s - {segment.end:05.1f}s] {assigned_speaker}: {text}"
        
        print(line)
        f.write(line + "\n")

print(f"\n✅ Pipeline Complete. Report saved to: {TRANSCRIPT_OUTPUT}")