import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import numpy as np
import json
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from starlette.responses import JSONResponse
from io import BytesIO
import base64

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

# Load the model (this will be replaced with your actual model loading code)
@st.cache_resource
def load_model():
    # Replace this with your actual model loading code
    # For example:
    # model = YourModelClass()
    # model.load_state_dict(torch.load('path/to/your/model.pth'))
    # model.eval()
    # return model
    
    # For now, we'll return a dummy model
    class DummyModel:
        def __init__(self):
            pass
        
        def predict(self, image):
            # This is a dummy prediction - replace with your actual model prediction
            confidences = {
                'akiec': 0.05,
                'bcc': 0.10,
                'bkl': 0.15,
                'df': 0.05,
                'mel': 0.20,
                'nv': 0.40,
                'vasc': 0.05
            }
            return 'nv', confidences
    
    return DummyModel()

# Preprocess image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# FastAPI endpoint for prediction
@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert('RGB')
        
        # Load model
        model = load_model()
        
        # Preprocess image
        # processed_image = preprocess_image(image)
        
        # Make prediction
        # In a real app, you would use the model to make a prediction
        # prediction, confidences = model(processed_image)
        
        # For now, we'll use the dummy model
        prediction, confidences = model.predict(image)
        
        # Generate some analysis details
        details = [
            "Image quality: Good",
            "Lesion border: Well-defined",
            "Color variation: Moderate"
        ]
        
        # Return prediction
        return JSONResponse(content={
            "prediction": prediction,
            "confidences": confidences,
            "details": details
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Streamlit UI (this will be shown when running with streamlit run api.py)
def main():
    st.title("DermaScan API Server")
    st.write("This is the backend API server for DermaScan.")
    st.write("The API is running at http://localhost:8501/api/predict")
    st.write("Users should interact with the frontend application, not this page.")
    
    # Add a simple test form for API testing
    st.header("API Testing")
    uploaded_file = st.file_uploader("Choose an image for testing", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Test prediction
        if st.button("Test Prediction"):
            with st.spinner("Analyzing..."):
                # Reset file pointer
                uploaded_file.seek(0)
                
                # Load model
                model = load_model()
                
                # Make prediction
                prediction, confidences = model.predict(image)
                
                # Display results
                st.success(f"Prediction: {prediction}")
                st.json(confidences)

# Run the FastAPI app with Streamlit
if __name__ == "__main__":
    import sys
    
    # Check if running with Streamlit
    if sys.argv[0].endswith("streamlit") or "streamlit" in sys.argv:
        main()
    else:
        # Run with uvicorn if not running with Streamlit
        uvicorn.run(app, host="0.0.0.0", port=8501)
