import os
import gc
from pathlib import Path
from dotenv import load_dotenv

# Strictly prevent silent OpenMP C++ crashes
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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

def clear_vram():
    """Forcefully clears GPU memory to protect the RTX 3050."""
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def extract_voice_embedding(audio_file_path: str) -> list:
    """
    Extracts a 512-dimensional voice biometric vector from an audio file.
    Immediately flushes VRAM after extraction to prevent OOM errors.
    """
    import torch
    from pyannote.audio import Model
    from pyannote.audio import Inference
    
    has_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if has_cuda else "cpu")
    
    try:
        # Load the raw Pyannote embedding model (loads to RAM/VRAM)
        model = Model.from_pretrained("pyannote/embedding", use_auth_token=hf_token)
        model.to(device)
        
        # Set window="whole" to aggregate the entire clip into ONE single 512-D vector
        inference = Inference(model, window="whole", device=device)
        
        # Generate the biometric fingerprint
        embedding_vector = inference(audio_file_path)
        
        # Convert the numpy array/tensor to a standard Python list (Required for ChromaDB)
        if hasattr(embedding_vector, "tolist"):
            vector_data = embedding_vector.tolist()
        else:
            vector_data = list(embedding_vector)
            
        return vector_data
        
    finally:
        # GUARANTEED VRAM FLUSH - Always executes before returning
        if 'inference' in locals():
            del inference
        if 'model' in locals():
            del model
        clear_vram()
