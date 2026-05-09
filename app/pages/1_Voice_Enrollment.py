import streamlit as st
import os
import chromadb
from pathlib import Path
import sys

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

col1, col2 = st.columns(2)
with col1:
    emp_name = st.text_input("Employee Full Name", placeholder="e.g., John Smith")
with col2:
    emp_role = st.text_input("Job Role", placeholder="e.g., Lead Architect")

st.markdown("### Audio Sample")
st.info("Please provide a 30 to 60-second clear voice sample of the employee speaking.")

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
        with st.spinner(f"Extracting 512-D Biometric Vector for {emp_name}..."):
            temp_path = DATA_DIR / "temp_enroll_sample.wav"
            
            try:
                # Save the audio buffer to a temporary physical file
                with open(temp_path, "wb") as f:
                    f.write(audio_file.read())
                
                # Extract the mathematical fingerprint using the newly refactored Engine
                engine = VoiceEmbeddingEngine()
                vector = engine.extract(str(temp_path))
                engine.cleanup()
                
                # Upsert into ChromaDB
                employee_id = emp_name.strip().lower().replace(" ", "_")
                
                collection.upsert(
                    ids=[employee_id],
                    embeddings=[vector],
                    metadatas=[{"name": emp_name.strip(), "role": emp_role.strip()}]
                )
                
                # Success notification without triggering st.rerun()
                st.success(f"Profile saved successfully! Voice biometric profile for **{emp_name}** ({emp_role}) is now securely stored in ChromaDB.")
                
            except Exception as e:
                st.error(f"Error during biometric extraction or database storage: {e}")
                
            finally:
                # Delete the temporary staging file from the hard drive
                if temp_path.exists():
                    os.remove(temp_path)
