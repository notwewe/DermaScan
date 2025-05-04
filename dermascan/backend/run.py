import subprocess
import sys
import os
import time
import threading
import webbrowser
import socket

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

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
    # Check if ports are already in use
    if is_port_in_use(8501):
        print("Warning: Port 8501 is already in use. Streamlit may not start correctly.")
    
    if is_port_in_use(8502):
        print("Warning: Port 8502 is already in use. API server may not start correctly.")
    
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
