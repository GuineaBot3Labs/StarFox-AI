import os
import subprocess
import time
import threading
from datetime import datetime

def ensure_directory_exists(folder_path):
    """Ensure the target directory exists."""
    os.makedirs(folder_path, exist_ok=True)

def capture_window_periodically(window_title, interval=4):
    """Capture the specified window at regular intervals."""
    ensure_directory_exists("Unlabeled")
    while True:
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"Unlabeled/screenshot_{timestamp}.png"
            # Adapt the command according to your system's screenshot utility
            cmd = f"import -window '{window_title}' '{filename}'"
            subprocess.run(cmd, shell=True, stderr=subprocess.DEVNULL)
            print(f"Screenshot saved to {filename}")
            time.sleep(interval)
        except Exception as e:
            print(f"Error capturing screenshot: {e}")

def start_capture_thread(window_title="Star Fox"):
    """Start the screenshot capture thread."""
    thread = threading.Thread(target=capture_window_periodically, args=(window_title,))
    thread.daemon = True  # Allows the script to exit even if the thread is running
    thread.start()

if __name__ == "__main__":
    window_title = "Star Fox (USA) (Rev 2)"  # Customize this to your game window title
    start_capture_thread(window_title)
    input("Press Enter to stop...\n")
