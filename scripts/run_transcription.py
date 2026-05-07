import os
from pathlib import Path
from faster_whisper import WhisperModel

# ==========================================
# DYNAMIC PATH ANCHORING
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
AUDIO_INPUT = str(DATA_DIR / "test_meeting.wav")

# 1. LOCK THE FOUNDATION (Protecting the C: Drive again)
WHISPER_MODEL_PATH = r"M:\Models\Whisper"
os.makedirs(WHISPER_MODEL_PATH, exist_ok=True)

print("=" * 50)
print("🚀 Booting up Faster-Whisper Engine...")

# 2. LOAD MODEL (Forces download to M: Drive)
# We are using the "base" model. It is incredibly fast and highly accurate for English.
# compute_type="float16" forces it to use your RTX 3050's optimized tensor cores.
model = WhisperModel("base", device="cuda", compute_type="float16", download_root=WHISPER_MODEL_PATH)
print("✅ Whisper Engine loaded to RTX GPU VRAM!")

# 3. RUN TRANSCRIPTION
print(f"🎧 Listening to: {AUDIO_INPUT}")
print("🧠 Transcribing audio (this takes a moment)...")

# beam_size=5 makes the AI double-check its work for better accuracy
segments, info = model.transcribe(AUDIO_INPUT, beam_size=5)

print("=" * 50)
print(f"Detected language: '{info.language}' with probability {info.language_probability:.2f}")
print("📝 TRANSCRIPTION TIMELINE:")
print("=" * 50)

# 4. PRINT THE RESULTS
for segment in segments:
    print(f"[{segment.start:05.1f}s - {segment.end:05.1f}s] --> {segment.text}")