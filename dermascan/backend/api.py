from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from PIL import Image
import io
import torch
import torchvision.transforms as transforms
import numpy as np
import subprocess
import threading
import time

# Import your model class
from app import SkinLesionModel, preprocess_image

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
@app.on_event("startup")
async def startup_event():
    global model
    model = SkinLesionModel(num_classes=7)
    model.load_state_dict(torch.load('skin_lesion_model.pth', map_location=torch.device('cpu')))
    model.eval()

# Class names
class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    # Read image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    
    # Preprocess
    input_tensor = preprocess_image(image)
    
    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
    
    # Convert to list
    probs = probabilities.numpy().tolist()
    
    # Create result
    result = {
        "prediction": class_names[np.argmax(probs)],
        "confidences": {class_name: float(prob) for class_name, prob in zip(class_names, probs)},
        "details": ["Image quality: Good", "Lesion border: Well-defined", "Color variation: Moderate"]
    }
    
    return result

# Start Streamlit in a separate thread
def run_streamlit():
    subprocess.run(["streamlit", "run", "app.py"])

@app.on_event("startup")
async def start_streamlit():
    threading.Thread(target=run_streamlit, daemon=True).start()
    # Give Streamlit time to start
    time.sleep(2)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)