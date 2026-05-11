import streamlit as st
import os
import chromadb
from pathlib import Path
import sys
import subprocess
import json
import logging

# Ensure the core module is accessible regardless of where Streamlit was launched
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from core.voice_biometrics import VoiceEmbeddingEngine

# Anchor the database to the data folder
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# Initialize a local persistent ChromaDB client
chroma_client = chromadb.PersistentClient(path=str(DATA_DIR / "chroma_db"))
collection = chroma_client.get_or_create_collection(
    name="employee_voices_v2",
    metadata={"hnsw:space": "cosine"}
)

st.set_page_config(page_title="Voice Enrollment", page_icon=":microphone:", layout="centered")

st.title(":microphone: Employee Voice Enrollment")
st.markdown("Register an employee's voice to automatically identify them in future meeting summaries instead of using generic tags like `SPEAKER_00`.")

# Display total count to verify database connection
total_enrolled = collection.count()
st.info(f"**Total Enrolled Employees in Database:** {total_enrolled}")

st.divider()

# Handle Redirects from Database Manager
if 'edit_name' in st.session_state:
    st.session_state.widget_emp_name = st.session_state['edit_name']
    del st.session_state['edit_name']
if 'edit_role' in st.session_state:
    st.session_state.widget_emp_role = st.session_state['edit_role']
    del st.session_state['edit_role']
if 'edit_original_id' in st.session_state:
    st.session_state['current_edit_id'] = st.session_state['edit_original_id']
    del st.session_state['edit_original_id']

col1, col2 = st.columns(2)
with col1:
    emp_name = st.text_input("Employee Full Name", key="widget_emp_name", placeholder="e.g., John Smith")
with col2:
    emp_role = st.text_input("Job Role", key="widget_emp_role", placeholder="e.g., Lead Architect")

st.markdown("### Audio Sample")
st.info("Please provide a 30 to 60-second clear voice sample of the employee speaking.")

with st.expander("📝 Need a script to read? Click here"):
    st.info("The sun shines brightly in the clear blue sky, warming the quiet town below. A gentle breeze rustles the green leaves on the tall oak trees. Dogs happily chase tennis balls across the soft grass of the local park. It is a perfect afternoon to sit outside, read a good book, and enjoy a warm cup of coffee.")

input_method = st.radio("Select Sample Source:", ["Upload Audio File", "Record from Microphone"], horizontal=True)

audio_file = None

if input_method == "Upload Audio File":
    audio_file = st.file_uploader("Upload Voice Sample", type=["wav", "mp3", "m4a", "flac"])
else:
    try:
        if "audio_key" not in st.session_state:
            st.session_state.audio_key = 0
            
        audio_file = st.audio_input("Record Voice Sample", key=f"mic_audio_widget_{st.session_state.audio_key}")
        
        # Display a Delete button if an audio file has been recorded
        if audio_file is not None:
            if st.button("🗑️ Delete Recording & Re-record", use_container_width=True):
                st.session_state.audio_key += 1
                st.rerun()
    except AttributeError:
        st.warning("Your version of Streamlit does not support native microphone recording. Please use the 'Upload Audio File' option.")

st.divider()

if st.button(":floppy_disk: Save Biometric Profile", type="primary", use_container_width=True):
    if not emp_name or not emp_role:
        st.error("Please provide both the Employee Name and Job Role.")
    elif not audio_file:
        st.error("Please provide a valid audio sample.")
    else:
        with st.spinner(f"Extracting Biometric Vector for {emp_name}..."):
            # Unique identifiers for temporary worker files
            safe_name = emp_name.strip().lower().replace(" ", "_")
            temp_raw = DATA_DIR / f"raw_enroll_{safe_name}.tmp"
            temp_wav = DATA_DIR / f"norm_enroll_{safe_name}.wav"
            task_json = DATA_DIR / f"tasks_enroll_{safe_name}.json"
            result_json = DATA_DIR / f"results_enroll_{safe_name}.json"
            
            try:
                # 1. Save raw upload/recording buffer
                with open(temp_raw, "wb") as f:
                    f.write(audio_file.read())
                
                # 2. Normalize audio using FFMPEG (16kHz, Mono, WAV)
                # This fixes the "Format not recognised" error for M4A/MP3 uploads
                subprocess.run([
                    "ffmpeg", "-y", "-i", str(temp_raw),
                    "-ac", "1", "-ar", "16000", str(temp_wav)
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

                if not temp_wav.exists():
                    raise RuntimeError("FFMPEG failed to create normalized WAV file.")

                # 3. Initialize tasks for the sandboxed worker
                tasks = [{"id": "enroll_task", "path": str(temp_wav)}]
                with open(task_json, 'w') as f:
                    json.dump(tasks, f)
                
                # 4. Spawn the isolated biometric worker process
                # This prevents Streamlit from losing connection during model load
                hf_token = os.getenv("HUGGINGFACE_TOKEN", "")
                worker_script = str(BASE_DIR / "core" / "biometric_worker.py")
                python_exe = sys.executable
                cmd = [python_exe, worker_script, str(task_json), str(result_json), hf_token]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    raise RuntimeError(f"Biometric worker crashed:\n{result.stderr}")

                if not result_json.exists():
                    raise RuntimeError("Biometric worker failed to generate result file.")
                    
                with open(result_json, 'r') as f:
                    res_data = json.load(f)
                
                vector = res_data.get("results", {}).get("enroll_task")
                if not vector:
                    raise RuntimeError("Biometric extraction failed (no vector returned).")
                
                # 5. Upsert into ChromaDB
                employee_id = safe_name
                
                # If we are updating an existing profile and the name (ID) changed, delete the old one
                if "current_edit_id" in st.session_state and st.session_state["current_edit_id"]:
                    old_id = st.session_state["current_edit_id"]
                    if old_id != employee_id:
                        try:
                            collection.delete(ids=[old_id])
                        except Exception:
                            pass
                    # Clear the edit state so future saves don't delete this ID
                    del st.session_state["current_edit_id"]

                collection.upsert(
                    ids=[employee_id],
                    embeddings=[vector],
                    metadatas=[{"name": emp_name.strip(), "role": emp_role.strip()}]
                )
                
                st.success(f"Profile saved successfully! Voice biometric profile for **{emp_name}** ({emp_role}) is now securely stored.")
                
            except Exception as e:
                st.error(f"Error during biometric extraction or database storage: {e}")
                logging.error(f"Enrollment Error: {str(e)}")
                
            finally:
                # 6. Comprehensive cleanup of all temporary worker files
                for p in [temp_raw, temp_wav, task_json, result_json]:
                    if p.exists():
                        try:
                            os.remove(p)
                        except:
                            pass
