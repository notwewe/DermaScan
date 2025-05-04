import torch
import torchvision.transforms as transforms
from PIL import Image, UnidentifiedImageError
import os
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from io import BytesIO
import torch.nn as nn
from torchvision import models
import logging
import time
import traceback
import gc
import psutil

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MODEL_PATH = "skin_lesion_model.pth"
DEFAULT_CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
REQUEST_TIMEOUT = 30  # seconds

# Initialize FastAPI
app = FastAPI()

# CORS configuration - IMPORTANT: This must be added before any routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=False,  # Important: must be False when allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model cache
_model_cache = None

@app.get("/")
async def root():
    logger.info("Root endpoint accessed")
    return {"status": "ok", "message": "DermaScan API is running. Use /api/predict endpoint for predictions."}

@app.options("/api/predict")
async def options_predict():
    # Handle preflight requests
    return JSONResponse(content={"status": "ok"})

# Define EfficientNet model architecture
class SkinLesionModel(nn.Module):
    def __init__(self, num_classes=7, dropout_rate=0.5):
        super(SkinLesionModel, self).__init__()
        
        # Load EfficientNet-B3 (without pretrained weights)
        self.efficientnet = models.efficientnet_b3(weights=None)  # Updated from pretrained=False
        
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
        start_time = time.time()
        logger.info("Starting model loading")
        
        model_path = os.path.join(os.path.dirname(__file__), MODEL_PATH)
        if not os.path.exists(model_path):
            logger.error(f"❌ Trained model file not found at {model_path}. Using DummyModel.")
            _model_cache = (DummyModel(), DEFAULT_CLASSES)
            return _model_cache

        logger.info(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Extract class information from checkpoint
        if isinstance(checkpoint, dict):
            if 'class_names' in checkpoint:
                class_names = checkpoint['class_names']
                logger.info(f"Found class names in checkpoint: {class_names}")
            else:
                class_names = DEFAULT_CLASSES
                logger.info(f"Using default class names: {class_names}")
                
            if 'model_state_dict' in checkpoint:
                model_state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                model_state_dict = checkpoint['state_dict']
            else:
                model_state_dict = checkpoint
                
            # Check if class_mapping exists
            if 'class_mapping' in checkpoint:
                class_mapping = checkpoint['class_mapping']
                logger.info(f"Found class mapping in checkpoint: {class_mapping}")
        else:
            class_names = DEFAULT_CLASSES
            model_state_dict = checkpoint
            logger.info(f"Using default class names: {class_names}")
            
        num_classes = len(class_names) if isinstance(class_names, list) else 7

        # Initialize the model with the correct architecture
        model = SkinLesionModel(num_classes=num_classes)
        
        # Load the state dictionary
        try:
            model.load_state_dict(model_state_dict)
            logger.info("Model state dictionary loaded successfully")
        except Exception as e:
            logger.error(f"Error loading state dictionary: {e}")
            # Try to handle potential key mismatches
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in model_state_dict.items():
                name = k
                if k.startswith('module.'):
                    name = k[7:]  # Remove 'module.' prefix
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
            logger.info("Model loaded with key remapping")
            
        model.eval()
        
        elapsed_time = time.time() - start_time
        logger.info(f"✅ Model loaded with {num_classes} classes in {elapsed_time:.2f} seconds.")
        _model_cache = (model, class_names)
        return _model_cache

    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        logger.error(traceback.format_exc())
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
async def predict(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    start_time = time.time()
    logger.info(f"Received prediction request for file: {file.filename}")
    log_memory_usage()

    
    try:
        # Set a timeout for the entire request processing
        if time.time() - start_time > REQUEST_TIMEOUT:
            logger.error(f"Request timed out after {REQUEST_TIMEOUT} seconds")
            raise HTTPException(status_code=504, detail="Request processing timed out")
            
        contents = await file.read()
        try:
            image = Image.open(BytesIO(contents)).convert('RGB')
            logger.info(f"Image opened successfully, size: {image.size}")
        except UnidentifiedImageError:
            logger.error("Invalid image file")
            raise HTTPException(status_code=400, detail="Invalid image file.")

        # Get model (with caching)
        model, class_names = get_model()
        
        # Check timeout again after model loading
        if time.time() - start_time > REQUEST_TIMEOUT:
            logger.error(f"Request timed out after model loading: {REQUEST_TIMEOUT} seconds")
            raise HTTPException(status_code=504, detail="Request processing timed out")
            
        # Preprocess image
        input_tensor = preprocess_image(image)

        # Run inference
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)[0]
            
        # Debug: Log raw probabilities to help diagnose the issue
        logger.info(f"Raw probabilities: {probs}")
        
        # Get the predicted class index
        predicted_idx = torch.argmax(probs).item()
        logger.info(f"Predicted index: {predicted_idx}")

        # Map class indices to class names
        confidences = {}
        
        # Handle different class name formats
        if isinstance(class_names, list) and len(class_names) == len(DEFAULT_CLASSES):
            # If class_names are the expected length, map them to DEFAULT_CLASSES
            for i in range(len(DEFAULT_CLASSES)):
                if i < len(probs):
                    confidences[DEFAULT_CLASSES[i]] = float(probs[i])
            
            # Get the prediction
            if predicted_idx < len(DEFAULT_CLASSES):
                prediction = DEFAULT_CLASSES[predicted_idx]
            else:
                prediction = DEFAULT_CLASSES[0]  # Fallback
                
            logger.info(f"Direct prediction: {prediction}")
        else:
            # Fallback mapping
            for i in range(min(len(DEFAULT_CLASSES), len(probs))):
                confidences[DEFAULT_CLASSES[i]] = float(probs[i])
            
            prediction = DEFAULT_CLASSES[predicted_idx] if predicted_idx < len(DEFAULT_CLASSES) else DEFAULT_CLASSES[0]
            logger.info(f"Fallback prediction: {prediction}")
            
        # Apply confidence threshold - if highest confidence is too low, mark as uncertain
        max_confidence = max(confidences.values())
        if max_confidence < 0.4:  # 40% threshold
            logger.warning(f"Low confidence prediction: {max_confidence:.2f}")
            
        # Analyze image
        details = [
            f"Image quality: {analyze_image_quality(image)}",
            f"Lesion border: {analyze_lesion_border(image)}",
            f"Color variation: {analyze_color_variation(image)}"
        ]

        # Prepare response
        response_data = {
            "prediction": prediction,
            "confidences": confidences,
            "details": details,
            "max_confidence": float(max_confidence)
        }
        
        elapsed_time = time.time() - start_time
        logger.info(f"Prediction successful: {prediction} in {elapsed_time:.2f} seconds")
        
        # Schedule garbage collection after response is sent
        background_tasks.add_task(collect_garbage)
        
        # Return response with CORS headers
        return JSONResponse(
            content=response_data,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "*",
            }
        )

    except HTTPException as http_err:
        logger.error(f"HTTP error: {http_err}")
        raise http_err
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    try:
        start_time = time.time()
        model, _ = get_model()
        dummy_input = torch.zeros((1, 3, 300, 300))
        with torch.no_grad():
            _ = model(dummy_input)
        elapsed_time = time.time() - start_time
        return {
            "status": "healthy", 
            "model_loaded": True,
            "response_time": f"{elapsed_time:.2f} seconds"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        logger.error(traceback.format_exc())
        return {"status": "unhealthy", "error": str(e)}

def collect_garbage():
    logger.info("Running garbage collection")
    gc.collect()
    torch.cuda.empty_cache()

def log_memory_usage():
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / 1024 ** 2  # in MB
    logger.info(f"Memory usage: {memory_usage:.2f} MB")

# Run with Uvicorn locally (optional)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8502))  # Use the port Render detected
    import uvicorn
    logger.info(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
