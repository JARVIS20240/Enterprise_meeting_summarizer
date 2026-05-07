import os
import torch
from pathlib import Path
from pyannote.audio import Pipeline
from dotenv import load_dotenv

# ==========================================
# DYNAMIC PATH ANCHORING
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
AUDIO_INPUT = str(DATA_DIR / "test_meeting.wav")

# 1. LOAD SECRETS FROM .ENV
# This looks for the .env file in the root directory
load_dotenv(BASE_DIR / ".env")
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# 2. LOCK THE FOUNDATION
CUSTOM_PATH = r"M:\Models\Voice_Separator"
os.environ["HF_HOME"] = CUSTOM_PATH
os.environ["TORCH_HOME"] = CUSTOM_PATH
os.environ["PYANNOTE_CACHE"] = CUSTOM_PATH

# 3. INJECT THE TOKEN (Now pulled safely from .env)
if not HF_TOKEN:
    print("❌ ERROR: HF_TOKEN not found in .env file!")
else:
    os.environ["HF_TOKEN"] = HF_TOKEN

print("=" * 50)
print("🚀 Initializing Voice Separator from M: Drive...")

# 4. LOAD PIPELINE
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
if torch.cuda.is_available():
    pipeline.to(torch.device("cuda")) 
    print("✅ Engine loaded to RTX GPU VRAM!")

# 5. RUN ANALYSIS
print(f"🎧 Processing: {AUDIO_INPUT}")
print("🧠 Analyzing voice patterns...")

diarization = pipeline(AUDIO_INPUT)

print("=" * 50)
print("📊 SPEAKER TIMELINE:")
print("=" * 50)

for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"[{turn.start:05.1f}s - {turn.end:05.1f}s] --> {speaker}")