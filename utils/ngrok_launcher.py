import os
import time
import subprocess
import threading
from pyngrok import ngrok
from dotenv import load_dotenv

# 0. THE ZOMBIE KILLER: Force quit any hidden ngrok processes (Windows version)
try:
    subprocess.run(["taskkill", "/f", "/im", "ngrok.exe"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2)
except Exception:
    pass

# 0. SETUP PATHS (Smart Location Detection)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DOTENV_PATH = os.path.join(ROOT_DIR, ".env")

# UPDATED PATH: Points to the new 'app' folder where the UI lives
APP_PATH = os.path.join(ROOT_DIR, "app", "Meeting_summary.py")

# 1. Load environment variables
load_dotenv(dotenv_path=DOTENV_PATH)
NGROK_TOKEN = os.getenv("NGROK_AUTH_TOKEN")

if not NGROK_TOKEN:
    print(f"ERROR: NGROK_AUTH_TOKEN not found in {DOTENV_PATH}", flush=True)
    exit(1)

# 2. Configure Ngrok
try:
    ngrok.set_auth_token(NGROK_TOKEN)
    print("SUCCESS: Ngrok token successfully loaded.", flush=True)
except Exception as e:
    print(f"ERROR: Failed to set Ngrok auth token: {e}", flush=True)
    exit(1)

# 3. Start Streamlit in a Background Process
def run_streamlit():
    # Use the specific python from the virtual environment
    python_exe = os.path.join(ROOT_DIR, "Enterprise", "Scripts", "python.exe")
    cmd = [python_exe, "-m", "streamlit", "run", APP_PATH, "--server.port", "8501", "--server.headless", "true", "--server.fileWatcherType", "none"]
    subprocess.run(cmd)

thread = threading.Thread(target=run_streamlit, daemon=True)
thread.start()

print("Waiting for Streamlit to initialize...", flush=True)
time.sleep(5) 

# 4. Generate the Public Web Link
try:
    # Kill any existing tunnels
    tunnels = ngrok.get_tunnels()
    for t in tunnels:
        ngrok.disconnect(t.public_url)
    
    public_url = ngrok.connect(8501)
    print("=" * 50, flush=True)
    print("Skyline Enterprise AI App is LIVE!", flush=True)
    print(f"Open this URL: {public_url.public_url}", flush=True)
    print("=" * 50, flush=True)
    
    # Keep the script running
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nClosing Ngrok tunnel...", flush=True)
    ngrok.kill()
except Exception as e:
    print(f"ERROR: Could not start Ngrok tunnel: {e}", flush=True)