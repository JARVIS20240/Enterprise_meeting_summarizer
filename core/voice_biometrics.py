import os
import gc
import torch
from pathlib import Path
from dotenv import load_dotenv

# Strictly prevent silent OpenMP C++ crashes
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import warnings
# Silence the deprecated torchaudio backend warning from pyannote.audio
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.core.io")

# ==========================================
# DYNAMIC PATH ANCHORING & MODEL CACHE
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# Enforce M: Drive model downloads to prevent C: Drive bloat
VOICE_MODEL_PATH = r"M:\Models\Voice_Separator"
os.environ["HF_HOME"] = VOICE_MODEL_PATH
os.environ["TORCH_HOME"] = VOICE_MODEL_PATH
os.environ["PYANNOTE_CACHE"] = VOICE_MODEL_PATH

hf_token = os.getenv("HUGGINGFACE_TOKEN")

class VoiceEmbeddingEngine:
    def __init__(self):
        """Initializes the Pyannote embedding model and pushes it to CUDA."""
        from pyannote.audio import Model
        
        has_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if has_cuda else "cpu")
        
        # Load the raw Pyannote embedding model
        self.model = Model.from_pretrained("pyannote/embedding", use_auth_token=hf_token)
        self.model.to(self.device)

    def extract(self, audio_file_path: str) -> list:
        """
        Extracts a 512-dimensional voice biometric vector from an audio file.
        Only performs the forward pass to prevent GIL locking during batch processing.
        """
        from pyannote.audio import Inference
        
        # Set window="whole" to aggregate the entire clip into ONE single 512-D vector
        inference = Inference(self.model, window="whole", device=self.device)
        
        # Generate the biometric fingerprint
        embedding_vector = inference(audio_file_path)
        
        # Convert the numpy array/tensor to a standard Python list (Required for ChromaDB)
        if hasattr(embedding_vector, "tolist"):
            vector_data = embedding_vector.tolist()
        else:
            vector_data = list(embedding_vector)
            
        return vector_data

    def cleanup(self):
        """Forcefully clears GPU memory to protect the RTX 3050."""
        if hasattr(self, 'model'):
            del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

if __name__ == "__main__":
    # Test Block to validate the Math Engine independently
    test_audio = str(BASE_DIR / "data" / "test_meeting.wav")
    
    print(f"Testing Biometric Extraction on: {test_audio}")
    if not os.path.exists(test_audio):
        print(f"Warning: {test_audio} does not exist. Please place a dummy audio file there to test.")
    else:
        try:
            engine = VoiceEmbeddingEngine()
            vec = engine.extract(test_audio)
            print("Extraction successful!")
            print(f"Extracted Vector Length: {len(vec)} (Expected 512)")
            print(f"First 5 numbers: {vec[:5]}")
            engine.cleanup()
        except Exception as e:
            print(f"Error during extraction: {e}")
