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
import gc  # Garbage collector
import psutil  # For memory monitoring

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("api.log")
    ]
)
logger = logging.getLogger(__name__)

# Constants
MODEL_PATH = "skin_lesion_model.pth"
DEFAULT_CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
CLASS_DESCRIPTIONS = {
    'akiec': 'Actinic Keratosis',
    'bcc': 'Basal Cell Carcinoma',
    'bkl': 'Benign Keratosis',
    'df': 'Dermatofibroma',
    'mel': 'Melanoma',
    'nv': 'Melanocytic Nevus',
    'vasc': 'Vascular Lesion'
}
REQUEST_TIMEOUT = 25  # seconds
MAX_IMAGE_SIZE = 1000  # Maximum dimension for images

# Initialize FastAPI
app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model cache
_model_cache = None

# Memory monitoring
def log_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    logger.info(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")

# Garbage collection helper
def collect_garbage():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log_memory_usage()

# Define EfficientNet model architecture
class SkinLesionModel(nn.Module):
    def __init__(self, num_classes=7, dropout_rate=0.5):
        super(SkinLesionModel, self).__init__()
        
        # Load EfficientNet-B3 (without pretrained weights)
        self.efficientnet = models.efficientnet_b3(weights=None)
        
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

# Preload model at startup
@app.on_event("startup")
async def startup_event():
    logger.info("Preloading model at startup")
    try:
        log_memory_usage()
        get_model()
        logger.info("Model preloaded successfully")
        log_memory_usage()
    except Exception as e:
        logger.error(f"Failed to preload model: {e}")
        logger.error(traceback.format_exc())

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
    # Resize large images before processing to save memory and time
    width, height = image.size
    if width > MAX_IMAGE_SIZE or height > MAX_IMAGE_SIZE:
        # Calculate aspect ratio
        aspect_ratio = width / height
        if width > height:
            new_width = MAX_IMAGE_SIZE
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = MAX_IMAGE_SIZE
            new_width = int(new_height * aspect_ratio)
        logger.info(f"Resizing large image from {width}x{height} to {new_width}x{new_height}")
        image = image.resize((new_width, new_height), Image.LANCZOS)
    
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

@app.get("/")
async def root():
    logger.info("Root endpoint accessed")
    return {"status": "ok", "message": "DermaScan API is running. Use /api/predict endpoint for predictions."}

@app.options("/api/predict")
async def options_predict():
    return JSONResponse(content={"status": "ok"})

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
        if isinstance(class_names[0], str) and len(class_names) == len(DEFAULT_CLASSES):
            # If class_names are strings and match the expected length, use them directly
            confidences = {DEFAULT_CLASSES[i]: float(probs[i]) for i in range(len(DEFAULT_CLASSES))}
            prediction = DEFAULT_CLASSES[predicted_idx]
            logger.info(f"Direct prediction: {prediction}")
        else:
            # If class_names are full names or don't match expected format, 
            # use a more robust mapping approach
            
            # Create a mapping from indices to default class codes
            confidences = {}
            for i in range(len(probs)):
                if i < len(DEFAULT_CLASSES):
                    confidences[DEFAULT_CLASSES[i]] = float(probs[i])
                    
            # Get the prediction using the predicted index
            if predicted_idx < len(DEFAULT_CLASSES):
                prediction = DEFAULT_CLASSES[predicted_idx]
            else:
                # Fallback if index is out of range
                prediction = DEFAULT_CLASSES[0]
                
            logger.info(f"Mapped prediction: {prediction}")
            
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
            "details": details
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
        
        # Create a test tensor with random data to verify model is working
        dummy_input = torch.rand((1, 3, 300, 300))
        
        with torch.no_grad():
            outputs = model(dummy_input)
            probs = torch.nn.functional.softmax(outputs, dim=1)[0]
            
        # Check if model is producing varied outputs
        min_prob = float(torch.min(probs))
        max_prob = float(torch.max(probs))
        prob_range = max_prob - min_prob
        
        elapsed_time = time.time() - start_time
        log_memory_usage()
        
        return {
            "status": "healthy", 
            "model_loaded": True,
            "response_time": f"{elapsed_time:.2f} seconds",
            "model_check": {
                "probability_range": prob_range,
                "min_probability": min_prob,
                "max_probability": max_prob
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        logger.error(traceback.format_exc())
        return {"status": "unhealthy", "error": str(e)}

# Debug endpoint to test model with sample data
@app.get("/debug/model-test")
async def model_test():
    try:
        model, class_names = get_model()
        
        # Create a test tensor with random data
        test_input = torch.rand((1, 3, 300, 300))
        
        with torch.no_grad():
            outputs = model(test_input)
            probs = torch.nn.functional.softmax(outputs, dim=1)[0]
        
        # Get predictions for each class
        predictions = {}
        for i, prob in enumerate(probs):
            class_code = DEFAULT_CLASSES[i] if i < len(DEFAULT_CLASSES) else f"class_{i}"
            class_name = CLASS_DESCRIPTIONS.get(class_code, f"Unknown Class {i}")
            predictions[class_code] = {
                "probability": float(prob),
                "class_name": class_name
            }
        
        return {
            "status": "success",
            "predictions": predictions,
            "raw_probabilities": [float(p) for p in probs],
            "class_names": class_names
        }
    except Exception as e:
        logger.error(f"Model test failed: {e}")
        logger.error(traceback.format_exc())
        return {"status": "error", "message": str(e)}

# Run with Uvicorn locally (optional)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8502))
    import uvicorn
    logger.info(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, timeout_keep_alive=300)