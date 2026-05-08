import sys
import os
import json
import traceback

# Strictly prevent silent OpenMP C++ crashes
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def run_diarization(wav_path, output_json, hf_token):
    try:
        import torch
        from pyannote.audio import Pipeline
        
        has_cuda = torch.cuda.is_available()
        
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
        if has_cuda:
            pipeline.to(torch.device("cuda"))
            
        diarization = pipeline(wav_path)
        
        segments = []
        # Convert Pyannote's proprietary tracks into a clean JSON serializable list
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker
            })
            
        with open(output_json, 'w') as f:
            json.dump(segments, f)
            
        sys.exit(0)
    except Exception as e:
        with open(output_json, 'w') as f:
            json.dump({"error": str(e), "traceback": traceback.format_exc()}, f)
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.exit("Usage: python pyannote_worker.py <wav_path> <output_json> <hf_token>")
        
    wav_path = sys.argv[1]
    output_json = sys.argv[2]
    hf_token = sys.argv[3]
    
    # Apply isolated environment paths to prevent redownloading
    VOICE_MODEL_PATH = r"M:\Models\Voice_Separator"
    os.environ["HF_HOME"] = VOICE_MODEL_PATH
    os.environ["TORCH_HOME"] = VOICE_MODEL_PATH
    os.environ["PYANNOTE_CACHE"] = VOICE_MODEL_PATH
    
    run_diarization(wav_path, output_json, hf_token)
