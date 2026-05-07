import subprocess
import time
from datetime import datetime

def monitor_vram():
    print("=" * 60)
    print(" 📊 SYSTEM TELEMETRY: RTX 3050 VRAM & CORE MONITORING")
    print("=" * 60)
    print("Timestamp    | Core Utilization | VRAM Allocation")
    print("-" * 60)
    
    try:
        while True:
            # Query the NVIDIA System Management Interface
            result = subprocess.check_output(
                [
                    'nvidia-smi',
                    '--query-gpu=utilization.gpu,memory.used,memory.total',
                    '--format=csv,noheader,nounits'
                ], 
                encoding='utf-8'
            )
            
            # Parse the metrics
            metrics = result.strip().split(', ')
            if len(metrics) == 3:
                gpu_util, mem_used, mem_total = metrics
                timestamp = datetime.now().strftime('%H:%M:%S')
                
                # Format output
                print(f"[{timestamp}] | Core: {gpu_util.rjust(3)}%         | VRAM: {mem_used.rjust(4)} MB / {mem_total} MB")
            
            time.sleep(1.0) # Refresh rate: 1 second
            
    except KeyboardInterrupt:
        print("\n[!] Telemetry sequence terminated by user.")
    except Exception as e:
        print(f"\n[!] Hardware interface failure: {e}")

if __name__ == "__main__":
    monitor_vram()