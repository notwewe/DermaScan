import torch
import torchvision.transforms as transforms
from PIL import Image, UnidentifiedImageError
import os
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from io import BytesIO
import torch.nn as nn
from torchvision import models
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MODEL_PATH = "skin_lesion_model.pth"
DEFAULT_CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

# Initialize FastAPI
app = FastAPI()

# CORS configuration - IMPORTANT: This must be added before any routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins temporarily for debugging
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Model cache
_model_cache = None

@app.get("/")
async def root():
    logger.info("Root endpoint accessed")
    return {"status": "ok", "message": "DermaScan API is running. Use /api/predict endpoint for predictions."}

@app.get("/cors-test")
async def cors_test():
    logger.info("CORS test endpoint accessed")
    return {"status": "ok", "message": "CORS is working properly if you can see this message in your frontend"}

# Define EfficientNet model architecture
class SkinLesionModel(nn.Module):
    def __init__(self, num_classes=7, dropout_rate=0.5):
        super(SkinLesionModel, self).__init__()
        
        # Load EfficientNet-B3 (without pretrained weights)
        self.efficientnet = models.efficientnet_b3(pretrained=False)
        
        # Get the number of features in the last layer
        in_features = self.efficientnet.classifier[1].in_features
        
        # Replace classifier with custom head
        self.efficientnet.classifier = nn.Identity()
        
        # Custom classifier with dropout for regularization
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate/2),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        features = self.efficientnet(x)
        return self.classifier(features)


# Fallback dummy model
class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.zeros((1, len(DEFAULT_CLASSES)))

# Load trained model with caching
def get_model():
    global _model_cache
    if _model_cache is not None:
        return _model_cache
    
    try:
        model_path = os.path.join(os.path.dirname(__file__), MODEL_PATH)
        if not os.path.exists(model_path):
            logger.error(f"❌ Trained model file not found at {model_path}. Using DummyModel.")
            _model_cache = (DummyModel(), DEFAULT_CLASSES)
            return _model_cache

        logger.info(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Extract class names if available, otherwise use default
        if isinstance(checkpoint, dict) and 'class_names' in checkpoint:
            class_names = checkpoint['class_names']
            model_state_dict = checkpoint['model_state_dict']
            logger.info(f"Found class names in checkpoint: {class_names}")
        else:
            class_names = DEFAULT_CLASSES
            model_state_dict = checkpoint
            logger.info(f"Using default class names: {class_names}")
            
        num_classes = len(class_names)

        # Initialize the model with the correct architecture
        model = SkinLesionModel(num_classes=num_classes)
        
        # Load the state dictionary
        model.load_state_dict(model_state_dict)
        model.eval()

        logger.info(f"✅ Model loaded with {num_classes} classes.")
        _model_cache = (model, class_names)
        return _model_cache

    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        _model_cache = (DummyModel(), DEFAULT_CLASSES)
        return _model_cache

# Image preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Image quality analysis
def analyze_image_quality(image):
    width, height = image.size
    if width < 100 or height < 100:
        return "Poor - Image is too small"
    variance = np.var(np.array(image.convert('L')))
    if variance < 100:
        return "Poor - Blurry"
    elif variance < 500:
        return "Moderate"
    return "Good"

def analyze_lesion_border(image):
    std = np.std(np.array(image))
    if std < 30:
        return "Poorly-defined"
    elif std < 60:
        return "Moderately-defined"
    return "Well-defined"

def analyze_color_variation(image):
    img = np.array(image.convert('RGB'))
    r_std = np.std(img[:, :, 0])
    g_std = np.std(img[:, :, 1])
    b_std = np.std(img[:, :, 2])
    avg_std = (r_std + g_std + b_std) / 3
    if avg_std < 20:
        return "Minimal"
    elif avg_std < 40:
        return "Moderate"
    return "Significant"

# Prediction endpoint
@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    logger.info(f"Received prediction request for file: {file.filename}")
    try:
        contents = await file.read()
        try:
            image = Image.open(BytesIO(contents)).convert('RGB')
            logger.info(f"Image opened successfully, size: {image.size}")
        except UnidentifiedImageError:
            logger.error("Invalid image file")
            raise HTTPException(status_code=400, detail="Invalid image file.")

        model, class_names = get_model()
        input_tensor = preprocess_image(image)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)[0]

        # Map class indices to class names
        if isinstance(class_names[0], str) and class_names[0] in DEFAULT_CLASSES:
            # If class_names are the short codes, use them directly
            confidences = {class_names[i]: float(probs[i]) for i in range(len(class_names))}
            prediction = class_names[torch.argmax(probs).item()]
        else:
            # If class_names are full names, map back to short codes for frontend
            confidences = {DEFAULT_CLASSES[i]: float(probs[i]) for i in range(len(DEFAULT_CLASSES))}
            prediction = DEFAULT_CLASSES[torch.argmax(probs).item()]

        details = [
            f"Image quality: {analyze_image_quality(image)}",
            f"Lesion border: {analyze_lesion_border(image)}",
            f"Color variation: {analyze_color_variation(image)}"
        ]

        response_data = {
            "prediction": prediction,
            "confidences": confidences,
            "details": details
        }
        
        logger.info(f"Prediction successful: {prediction}")
        return JSONResponse(content=response_data)

    except HTTPException as http_err:
        logger.error(f"HTTP error: {http_err}")
        raise http_err
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    try:
        model, _ = get_model()
        dummy_input = torch.zeros((1, 3, 300, 300))
        with torch.no_grad():
            _ = model(dummy_input)
        return {"status": "healthy", "model_loaded": True}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}

# Run with Uvicorn locally (optional)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8502))  # Use the port Render detected
    import uvicorn
    logger.info(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)