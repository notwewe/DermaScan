import subprocess
import sys
import os
import time
import threading
import webbrowser

def run_api_server():
    print("Starting API server...")
    subprocess.run([sys.executable, "api.py"])

def run_streamlit_app():
    print("Starting Streamlit app...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"])

def open_browser():
    # Wait a bit for Streamlit to start
    time.sleep(5)
    # Open browser
    webbrowser.open("http://localhost:8501")

if __name__ == "__main__":
    # Start API server in a separate thread
    api_thread = threading.Thread(target=run_api_server)
    api_thread.daemon = True
    api_thread.start()
    
    # Wait a bit for the API server to start
    time.sleep(2)
    
    # Open browser in a separate thread
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Run Streamlit app in the main thread
    run_streamlit_app()
