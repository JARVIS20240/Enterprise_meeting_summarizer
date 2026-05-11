import sys
import os
import json
import traceback
import torch

# Strictly prevent silent OpenMP C++ crashes
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import warnings
# Silence the deprecated torchaudio backend warning from pyannote.audio
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.core.io")

def run_extraction(input_json, output_json, hf_token):
    try:
        from core.voice_biometrics import VoiceEmbeddingEngine
        
        with open(input_json, 'r') as f:
            tasks = json.load(f)
            
        engine = VoiceEmbeddingEngine()
        results = {}
        
        for task in tasks:
            task_id = task["id"]
            path = task["path"]
            if os.path.exists(path):
                vector = engine.extract(path)
                results[task_id] = vector
            else:
                results[task_id] = None
                
        engine.cleanup()
        
        with open(output_json, 'w') as f:
            json.dump({"results": results}, f)
            
        sys.exit(0)
    except Exception as e:
        with open(output_json, 'w') as f:
            json.dump({"error": str(e), "traceback": traceback.format_exc()}, f)
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.exit("Usage: python biometric_worker.py <input_json> <output_json> <hf_token>")
        
    input_json = sys.argv[1]
    output_json = sys.argv[2]
    hf_token = sys.argv[3]
    
    # Apply isolated environment paths to prevent redownloading
    VOICE_MODEL_PATH = r"M:\Models\Voice_Separator"
    os.environ["HF_HOME"] = VOICE_MODEL_PATH
    os.environ["TORCH_HOME"] = VOICE_MODEL_PATH
    os.environ["PYANNOTE_CACHE"] = VOICE_MODEL_PATH
    
    # Ensure current directory is in path for relative imports
    sys.path.append(os.getcwd())
    
    run_extraction(input_json, output_json, hf_token)
