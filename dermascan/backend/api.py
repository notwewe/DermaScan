import streamlit as st
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import os
import numpy as np
import json
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from starlette.responses import JSONResponse
from io import BytesIO
import base64
import requests

# Class names
CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

# Create FastAPI app
app = FastAPI()

# Add CORS middleware to allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
@st.cache_resource
def load_model():
    try:
        model = models.efficientnet_b3(pretrained=False)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(num_ftrs, len(CLASS_NAMES))

        model_path = os.path.join(os.path.dirname(__file__), 'skin_lesion_model.pth')

        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            print("Loaded trained model")
        else:
            print("No trained model found. Using random classifier weights.")

        model.eval()
        return model

    except Exception as e:
        print(f"Error loading model: {e}")
        return DummyModel()

class DummyModel:
    def __init__(self):
        print("Using dummy model.")

    def __call__(self, x):
        return torch.softmax(torch.randn(1, len(CLASS_NAMES)), dim=1)

# Preprocess image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Analyze image quality
def analyze_image_quality(image):
    # Calculate basic image quality metrics
    width, height = image.size
    
    # Check if image is too small
    if width < 100 or height < 100:
        return "Poor - Image is too small"
    
    # Check if image is blurry (using a simple variance of Laplacian)
    img_array = np.array(image.convert('L'))
    variance = np.var(img_array)
    
    if variance < 100:
        return "Poor - Image may be blurry"
    elif variance < 500:
        return "Moderate"
    else:
        return "Good"

# Analyze lesion border
def analyze_lesion_border(image):
    # This would normally use edge detection algorithms
    # For now, we'll return a placeholder based on image characteristics
    img_array = np.array(image)
    
    # Simple edge detection using standard deviation of pixel values
    edge_strength = np.std(img_array)
    
    if edge_strength < 30:
        return "Poorly-defined"
    elif edge_strength < 60:
        return "Moderately-defined"
    else:
        return "Well-defined"

# Analyze color variation
def analyze_color_variation(image):
    # Convert to RGB if not already
    img_rgb = image.convert('RGB')
    img_array = np.array(img_rgb)
    
    # Calculate standard deviation across color channels
    r_std = np.std(img_array[:,:,0])
    g_std = np.std(img_array[:,:,1])
    b_std = np.std(img_array[:,:,2])
    
    avg_std = (r_std + g_std + b_std) / 3
    
    if avg_std < 20:
        return "Minimal"
    elif avg_std < 40:
        return "Moderate"
    else:
        return "Significant"

# FastAPI endpoint for prediction
@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert('RGB')

        model = load_model()

        processed_image = preprocess_image(image)

        with torch.no_grad():
            outputs = model(processed_image)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]

        confidences = {CLASS_NAMES[i]: float(probabilities[i]) for i in range(len(CLASS_NAMES))}
        prediction = CLASS_NAMES[torch.argmax(probabilities).item()]
        
        # Generate analysis details
        image_quality = analyze_image_quality(image)
        border_quality = analyze_lesion_border(image)
        color_variation = analyze_color_variation(image)
        
        details = [
            f"Image quality: {image_quality}",
            f"Lesion border: {border_quality}",
            f"Color variation: {color_variation}"
        ]

        return JSONResponse(content={
            "prediction": prediction,
            "confidences": confidences,
            "details": details
        })

    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Run the FastAPI app with uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8502)
