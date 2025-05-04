import requests
import sys
import time

def test_api_connection():
    """Test if the API server is running and responding"""
    print("Testing API connection...")
    
    # Try to connect to the root endpoint
    try:
        response = requests.get("https://dermascan-56zs.onrender.com")
        if response.status_code == 200:
            print("✅ API server is running and responding")
            print(f"Response: {response.json()}")
            return True
        else:
            print(f"❌ API server returned status code {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API server")
        return False

def main():
    # Try to connect to the API server
    success = test_api_connection()
    
    if not success:
        print("\nThe API server is not running or not responding.")
        print("Please make sure to run 'python api.py' in a separate terminal.")
        sys.exit(1)
    
    print("\nAPI server is ready to accept requests.")

if __name__ == "__main__":
    main()
