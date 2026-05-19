import os
import warnings
import base64

# This strictly prevents the silent OpenMP C++ crash on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Suppress the torchaudio deprecation warning from pyannote.audio
warnings.filterwarnings("ignore", message="torchaudio._backend.set_audio_backend has been deprecated")

import logging
import streamlit as st
import re
import requests
import tempfile
import json
import subprocess
import sys
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pdfkit
import markdown
import gc

# ==========================================
# DYNAMIC PATH ANCHORING & IMPORTS
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
LOG_PATH = str(BASE_DIR / "system_debug.log")

# Add the root directory to the system path so we can import the core engine securely
sys.path.append(str(BASE_DIR))
from core.merger_engine import transcribe_audio

# ══════════════════════════════════════════════════════════
# 0. GLOBAL LOGGING ARCHITECTURE
# ══════════════════════════════════════════════════════════
logging.basicConfig(
    filename=LOG_PATH,
    filemode='a',
    format='[%(asctime)s] %(levelname)s - %(message)s',
    level=logging.INFO
)
logging.info("=== STREAMLIT SERVER INITIALIZED (OPTIMIZED VRAM) ===")

# ══════════════════════════════════════════════════════════
# CONSTANTS & SETTINGS
# ══════════════════════════════════════════════════════════
OLLAMA_API = "http://localhost:11434/api/generate"
WORKER_MODEL = "gemma4:e2b"
EXECUTIVE_MODEL = "gemma4:e2b"

SUPPORTED_VIDEO  = [".mp4", ".mov", ".avi", ".mkv"]
SUPPORTED_AUDIO  = [".mp3", ".wav", ".m4a", ".ogg", ".flac"]

# ══════════════════════════════════════════════════════════
# 1. AUDIO EXTRACTION
# ══════════════════════════════════════════════════════════
def prepare_audio(uploaded_file=None, local_path=None) -> str:
    logging.info("Starting ultra-fast audio extraction...")
    tmp_dir = tempfile.mkdtemp()
    
    if uploaded_file:
        suffix = Path(uploaded_file.name).suffix.lower()
        raw_path = os.path.join(tmp_dir, f"input{suffix}")
        with open(raw_path, "wb") as f:
            f.write(uploaded_file.read())
    else:
        raw_path = local_path
        suffix = Path(raw_path).suffix.lower()

    wav_path = os.path.join(tmp_dir, "audio.wav")

    try:
        command = [
            "ffmpeg", "-y", "-i", raw_path, 
            "-ac", "1", "-ar", "16000", "-vn", 
            wav_path
        ]
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        logging.info(f"FFMPEG extraction successful: {wav_path}")
    except Exception as e:
        logging.error(f"FFMPEG extraction failed: {e}. Falling back to pydub/moviepy...")
        if suffix in SUPPORTED_VIDEO:
            from moviepy.editor import VideoFileClip
            clip = VideoFileClip(raw_path)
            clip.audio.write_audiofile(wav_path, verbose=False, logger=None)
            clip.close()
        else:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(raw_path)
            audio = audio.set_channels(1).set_frame_rate(16000)
            audio.export(wav_path, format="wav")
            
    return wav_path

# ══════════════════════════════════════════════════════════
# 2. THE MULTI-MODEL MAP-REDUCE ENGINE
# ══════════════════════════════════════════════════════════
def call_ollama_stream(model_name: str, prompt: str, keep_alive: str = "5m", container=None):
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": True,
        "keep_alive": keep_alive,
        "options": {
            "num_ctx": 8192
        }
    }
    full_text = ""
    try:
        response = requests.post(OLLAMA_API, json=payload, stream=True)
        if response.status_code == 200:
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    data = json.loads(decoded_line)
                    if "response" in data:
                        chunk = data["response"]
                        full_text += chunk
                        if container:
                            container.markdown(full_text + "▌")
            if container:
                container.markdown(full_text)
        return full_text
    except Exception as e:
        error_msg = f"[Error: Could not connect to Ollama. Is it running? {str(e)}]"
        logging.error(f"Ollama connection failed: {e}")
        if container:
            container.error(error_msg)
        return error_msg

def process_transcript(transcript: str, progress_bar, status_text, language: str) -> str:
    logging.info("Starting process_transcript map-reduce engine...")
    
    # --- STEP 1: CHUNKING (MAP PHASE) ---
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=25000, 
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(transcript)
    total_chunks = len(chunks)

    extracted_notes = []
    for i, chunk in enumerate(chunks):
        status_text.write(f":mag: Analysis Engine ({WORKER_MODEL}) processing segment {i+1} of {total_chunks}...")
        logging.info(f"Processing chunk {i+1}/{total_chunks}")

        map_prompt = f"""
        Read the following meeting transcript chunk. The transcript includes generic speaker tags like "SPEAKER_00:", "SPEAKER_01:", etc.
        Output ONLY a concise, bulleted list of raw facts, decisions, and action items.

        STRICT RULES:
        1. DEDUCE NAMES: Try to map the generic tags (like SPEAKER_00) to real names based on context (e.g. if SPEAKER_00 says "Hi, I'm John", map SPEAKER_00 to John). Use these deduced names in your output instead of the generic tags whenever possible.
        2. EXTREME SHORTHAND: Write as little text as possible. Do not write full paragraphs.
        3. NO FLUFF: Ignore pleasantries and conversational filler.
        4. SOFT ATTRIBUTION: Always note WHO said WHAT using their deduced name or their generic tag if the name is unknown (e.g., "John: ... " or "SPEAKER_01: ... ").
        5. EXTRACT METRICS (IF PRESENT): Capture specific numbers, deadlines, or timelines.
        6. EXTRACT BUSINESS RISKS (IF PRESENT): Note any discussions around legal liability, document approval statuses, security severities, or compliance.
        7. ACTION ITEMS / OWNER: Identify the speaker or owner of a task. Use their real name if deduced, otherwise use the generic tag. Only use "Team" if absolutely no one is specified.
        8. LANGUAGE: Write all your notes strictly in English. Do not translate to English unless the meeting was in English.

        Transcript Chunk:
        {chunk}

        Raw Shorthand Notes:
        """

        
        stream_container = st.empty()
        notes = call_ollama_stream(WORKER_MODEL, map_prompt, keep_alive="5m", container=stream_container)
        extracted_notes.append(notes)
        stream_container.empty()
        progress_bar.progress((i + 1) / total_chunks)

    logging.info("Optimizing memory, flushing worker model state...")
    requests.post(OLLAMA_API, json={"model": WORKER_MODEL, "keep_alive": 0})

    debug_path = str(DATA_DIR / "DEBUG_notes_dump.txt")
    with open(debug_path, "w", encoding="utf-8") as f:
        f.write(f"TOTAL EXPECTED CHUNKS: {total_chunks}\n")
        f.write(f"TOTAL CAPTURED CHUNKS: {len(extracted_notes)}\n")
        f.write("="*40 + "\n\n")
        for idx, note in enumerate(extracted_notes):
            f.write(f"--- CHUNK {idx + 1} ---\n")
            f.write(str(note) + "\n\n")
            
    logging.info(f"DEBUG: Saved {len(extracted_notes)} out of {total_chunks} chunks to {debug_path}")

    status_text.write(f":brain: Synthesis Engine ({EXECUTIVE_MODEL}) generating executive report...")
    combined_notes = "\n\n".join(extracted_notes)

    reduce_prompt = f"""
    Synthesize the extracted shorthand notes into a final, professional executive report.

    CRITICAL RULES:
    1. ANTI-HALLUCINATION: You must ONLY use the information provided in the "Extracted Shorthand Notes" below. Do NOT invent fake Service Level Agreements (SLAs), threat levels, or names.
    2. ATTRIBUTION: You MUST identify who is responsible for action items. Replace the template placeholder with the ACTUAL name or tag found in the notes.
    3. LANGUAGE: Write the final executive report strictly in English. Do not translate to English unless the notes are in English.
    4. SUGGESTED TOPICS FOR FUTURE MEETINGS (AI GENERATION): You must act as a strategic consultant. Analyze the highly relevant themes of this meeting and GENERATE 2 to 3 logical, strategic follow-up topics that the team should explore in their next meeting to progress the conversation.

    You MUST copy the EXACT structure below. Fill in the information based ONLY on what actually happened in the meeting.

    Overview
    This meeting focused on the following areas:
    - [Topic 1]
    - [Topic 2]

    Discussion Points
    1. [Major Topic 1]
    - [Point 1]
    - [Point 2]
    2. [Major Topic 2]
    - [Point 1]

    Action Items
    1. **[Short Action Title]** - [Speaker Name or 'Team']: [Specific Task Details]
    2. **[Short Action Title]** - [Speaker Name or 'Team']: [Specific Task Details]

    Suggested Topics for Future Meetings
    - [Future Topic 1]
    - [Future Topic 2]

    Extracted Shorthand Notes:
    {combined_notes}
    """


    exec_container = st.empty()
    logging.info("Executing executive summary generation...")
    final_summary = call_ollama_stream(EXECUTIVE_MODEL, reduce_prompt, keep_alive=0, container=exec_container)

    logging.info("Executive summary generation complete.")
    exec_container.empty()
    return final_summary

# ══════════════════════════════════════════════════════════
# 3. PDF GENERATION (ReportLab Architecture)
# ══════════════════════════════════════════════════════════
def generate_pdf(summary: str, duration_sec: float) -> str:
    logging.info("Initializing HTML/CSS PDF generation sequence...")
    try:
        out_path = os.path.join(tempfile.mkdtemp(), "refined_meeting_summary.pdf")
        
        mins = int(duration_sec // 60)
        secs = int(duration_sec % 60)
        
        summary = re.sub(r'^[ \t]+', '', summary, flags=re.MULTILINE)
        summary = summary.replace('•', '-').replace('–', '-').replace('—', '-')
        summary = re.sub(r'^-\s*', '- ', summary, flags=re.MULTILINE)
        
        for header in ["Overview", "Discussion Points", "Action Items", "Suggested Topics for Future Meetings"]:
            summary = re.sub(rf'^\s*({header})\s*$', rf'## \1', summary, flags=re.MULTILINE)
            
        summary = re.sub(r'^(\d+\.\s+[A-Za-z][^\*].*?)$', r'### \1', summary, flags=re.MULTILINE)
        summary = re.sub(r'([^\s])\s*\n(-|\*) ', r'\1\n\n\2 ', summary)
        summary = re.sub(r'([^\s])\s*\n(\d+\.) ', r'\1\n\n\2 ', summary)

        html_content = markdown.markdown(summary, extensions=['extra', 'sane_lists'])
        html_content = html_content.replace('<h2>', '<hr class="section-divider"><h2>')
        
        def convert_action_list(list_content):
            list_content = re.sub(r'(<strong>.*?</strong>\s*-\s*)([A-Za-z0-9\s\(\)\'\.\_]+?):', r'\1<span class="owner">\2</span>:', list_content)
            boxes = re.sub(r'<li>(.*?)</li>', r'<li class="action-item-box">\1</li>', list_content, flags=re.DOTALL)
            return boxes
            
        pattern = r'(<h2>Action Items</h2>(?:(?!<h2).)*?)<(ul|ol)>(.*?)</\2>'
        html_content = re.sub(pattern, 
                              lambda m: m.group(1) + '<ol class="action-items">' + convert_action_list(m.group(3)) + '</ol>', 
                              html_content, flags=re.DOTALL)
        
        html_template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <style>
                body {{
                    font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
                    color: #333333;
                    line-height: 1.6;
                    margin: 0;
                    padding: 0;
                    font-size: 15px;
                }}
                .main-title {{
                    color: #0f2555;
                    font-size: 32px;
                    font-weight: 800;
                    margin-top: 0;
                    margin-bottom: 20px;
                }}
                .meta-box {{
                    background-color: #f1f5f9;
                    padding: 14px 20px;
                    border-radius: 6px;
                    display: table;
                    width: 100%;
                    box-sizing: border-box;
                    margin-bottom: 25px;
                }}
                .meta-left {{
                    display: table-cell;
                    text-align: left;
                    font-size: 14px;
                    color: #333;
                }}
                .meta-right {{
                    display: table-cell;
                    text-align: right;
                    font-size: 14px;
                    color: #333;
                }}
                hr.section-divider {{
                    border: none;
                    border-top: 1px solid #e2e8f0;
                    margin: 25px 0 20px 0;
                }}
                h2 {{
                    color: #0f2555;
                    font-size: 24px;
                    font-weight: 700;
                    margin-top: 0;
                    margin-bottom: 18px;
                    position: relative;
                    padding-left: 14px;
                    page-break-after: avoid;
                }}
                h2::before {{
                    content: "";
                    position: absolute;
                    left: 0;
                    top: 2px;
                    bottom: 2px;
                    width: 4px;
                    background-color: #1d4ed8;
                    border-radius: 2px;
                }}
                h3 {{
                    color: #475569;
                    font-size: 18px;
                    font-weight: 600;
                    margin: 15px 0 10px 0;
                    page-break-after: avoid;
                }}
                ul, ol {{
                    margin-top: 0;
                    margin-bottom: 15px;
                    padding-left: 20px;
                }}
                li {{
                    margin-bottom: 8px;
                }}
                .action-items {{
                    margin-top: 15px;
                    padding-left: 0;
                }}
                .action-item-box {{
                    background-color: #f8fafc;
                    border-left: 3px solid #f59e0b;
                    padding: 12px 15px;
                    margin-bottom: 12px;
                    border-radius: 0 4px 4px 0;
                    font-size: 14px;
                    page-break-inside: avoid;
                    list-style-position: inside;
                    font-weight: 600;
                }}
                .action-item-box p {{
                    margin: 0;
                    display: inline;
                    font-weight: 400;
                }}
                .owner {{
                    color: #1d4ed8;
                    font-weight: 700;
                }}
            </style>
        </head>
        <body>
            <h1 class="main-title">Refined Meeting Summary</h1>
            
            <div class="meta-box">
                <div class="meta-left">
                    <strong>Meeting Duration:</strong> {mins} minutes, {secs} seconds
                </div>
                <div class="meta-right">
                    <strong>Generated by:</strong> Enterprise Briefing AI
                </div>
            </div>
            
            <div class="content">
                {html_content}
            </div>
        </body>
        </html>
        """
        
        options = {
            'enable-local-file-access': '',
            'margin-top': '20mm',
            'margin-right': '20mm',
            'margin-bottom': '25mm',
            'margin-left': '20mm',
            'encoding': 'UTF-8',
            'footer-center': 'Confidential - For Internal Use Only | Generated by AI Synthesis Engine',
            'footer-font-name': 'Segoe UI',
            'footer-font-size': '8',
            'footer-spacing': '5'
        }
        
        path_wkhtmltopdf = r'D:\Softwares\wkhtmltopdf\bin\wkhtmltopdf.exe'
        if os.path.exists(path_wkhtmltopdf):
            config = pdfkit.configuration(wkhtmltopdf=path_wkhtmltopdf)
            pdfkit.from_string(html_template, out_path, configuration=config, options=options)
        else:
            pdfkit.from_string(html_template, out_path, options=options)
            
        logging.info("HTML/CSS PDF generated successfully.")
        return out_path
        
    except Exception as e:
        logging.critical(f"FATAL ERROR IN HTML PDF GENERATION: {str(e)}", exc_info=True)
        raise e

# ══════════════════════════════════════════════════════════
# 4. STREAMLIT UI
# ══════════════════════════════════════════════════════════
def main():
    st.set_page_config(page_title="Enterprise Briefing AI", page_icon=":bookmark_tabs:", layout="centered")

    st.title(":bookmark_tabs: Enterprise Meeting Summarizer (GPU Accelerated)")
    st.divider()

    if "transcript" not in st.session_state:
        st.session_state.transcript = None
    if "duration" not in st.session_state:
        st.session_state.duration = 0
    if "language" not in st.session_state:
        st.session_state.language = "en"
    if "pdf_path" not in st.session_state:
        st.session_state.pdf_path = None
    if "summary" not in st.session_state:
        st.session_state.summary = None
    if "current_file" not in st.session_state:
        st.session_state.current_file = None

    input_method = st.radio("Select Audio Source:", ["Server Path (Fastest)", "Web Interface Upload"], horizontal=True)

    audio_path_to_process = None
    filename_display = None
    uploaded_file_obj = None

    if input_method == "Web Interface Upload":
        uploaded = st.file_uploader(":open_file_folder: Upload Media", type=["mp4","mov","mp3","wav","m4a"])
        if uploaded:
            audio_path_to_process = "web_upload"
            filename_display = uploaded.name
            uploaded_file_obj = uploaded
    else:
        local_path = st.text_input(":open_file_folder: Enter Absolute Server Path (e.g., /content/audio.mp3):")
        if local_path and os.path.exists(local_path):
            audio_path_to_process = local_path
            filename_display = os.path.basename(local_path)

    if audio_path_to_process:
        
        if st.session_state.current_file != filename_display:
            st.session_state.transcript = None
            st.session_state.duration = 0
            st.session_state.language = "en"
            st.session_state.pdf_path = None
            st.session_state.summary = None
            st.session_state.current_file = filename_display

        if st.session_state.transcript is None:
            if st.button(":arrow_forward: Initialize Transcription", type="primary", use_container_width=True):
                st.markdown("### :gear: Processing media stream...")
                status_container = st.empty()
                status_container.info("Extracting audio channels...")
                
                if audio_path_to_process == "web_upload":
                    wav_path = prepare_audio(uploaded_file=uploaded_file_obj)
                else:
                    wav_path = prepare_audio(local_path=audio_path_to_process)

                live_text = st.empty()
                
                # This calls your new Engine directly!
                transcript, duration, language = transcribe_audio(wav_path, live_status_container=live_text)
                
                
                st.session_state.transcript = transcript
                st.session_state.duration = duration
                st.session_state.language = language
                live_text.empty() 
                status_container.success("Transcription & Diarization complete. VRAM freed.")

    if st.session_state.transcript is not None:
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            b64_transcript = base64.b64encode(st.session_state.transcript.encode()).decode()
            download_html = f'''
                <a href="data:file/txt;base64,{b64_transcript}" download="{filename_display}_transcript.txt" 
                   style="display: inline-block; padding: 0.5rem 1rem; background-color: #262730; 
                          color: white; border-radius: 0.5rem; text-decoration: none; 
                          border: 1px solid rgba(250, 250, 250, 0.2); width: 100%; text-align: center;">
                   📥 Download Raw Transcript (.txt)
                </a>
            '''
            st.markdown(download_html, unsafe_allow_html=True)
            
        with col2:
            start_summary = st.button("Initialize AI Summary", type="primary", use_container_width=True)

        if start_summary:
            st.divider()
            st.markdown("### :brain: Engaging AI Synthesis Engine...")
            progress_bar = st.progress(0)
            status_text = st.empty()

            summary = process_transcript(st.session_state.transcript, progress_bar, status_text, st.session_state.language)
            st.session_state.summary = summary
            
            # 🚨 RESTORED: This is the code that was missing to generate the PDF! 🚨
            status_text.markdown("**:triangular_ruler: Formatting PDF...**")
            try:
                pdf_path = generate_pdf(summary, st.session_state.duration)
                st.session_state.pdf_path = pdf_path
            except Exception as e:
                st.error(f"PDF Generation Failed. Check system_debug.log.")
                logging.error("Continuing application state despite PDF failure.")

            status_text.success(":white_check_mark: Pipeline finalized.")

    if st.session_state.pdf_path is not None or st.session_state.summary is not None:
        st.divider()
        st.success(":tada: Artifacts Ready")
        
        if st.session_state.pdf_path is not None:
            with open(st.session_state.pdf_path, "rb") as f:
                pdf_bytes = f.read()
            st.download_button("Download Executive Report (PDF)", pdf_bytes, "executive_meeting_summary.pdf", mime="application/pdf", use_container_width=True, type="primary")

        if st.session_state.summary is not None:
            with st.expander("Inspect Raw Summary Output"):
                st.markdown(st.session_state.summary)

if __name__ == "__main__":
    try:
        main()
    except Exception as fatal_error:
        logging.critical(f"UNHANDLED STREAMLIT CRASH: {str(fatal_error)}", exc_info=True)
        st.error("A fatal system error occurred. Please check the logs.")