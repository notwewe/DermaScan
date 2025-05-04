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

# Constants
MODEL_PATH = "skin_lesion_model.pth"
DEFAULT_CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

# Initialize FastAPI
app = FastAPI()

# Allow CORS for Vercel frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://derma-scan-kappa.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"status": "ok", "message": "DermaScan API is running. Use /api/predict endpoint for predictions."}


# Fallback dummy model
class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.zeros((1, len(DEFAULT_CLASSES)))

# Load trained model
def load_model():
    try:
        model_path = os.path.join(os.path.dirname(__file__), MODEL_PATH)
        if not os.path.exists(model_path):
            print("❌ Trained model file not found. Using DummyModel.")
            return DummyModel(), DEFAULT_CLASSES

        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        class_names = checkpoint.get('class_names', DEFAULT_CLASSES)
        num_classes = len(class_names)

        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        print(f"✅ Model loaded with {num_classes} classes.")
        return model, class_names

    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return DummyModel(), DEFAULT_CLASSES

# Image preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
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
    try:
        contents = await file.read()
        try:
            image = Image.open(BytesIO(contents)).convert('RGB')
        except UnidentifiedImageError:
            raise HTTPException(status_code=400, detail="Invalid image file.")

        model, class_names = load_model()
        input_tensor = preprocess_image(image)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)[0]

        confidences = {class_names[i]: float(probs[i]) for i in range(len(class_names))}
        prediction = class_names[torch.argmax(probs).item()]

        details = [
            f"Image quality: {analyze_image_quality(image)}",
            f"Lesion border: {analyze_lesion_border(image)}",
            f"Color variation: {analyze_color_variation(image)}"
        ]

        return JSONResponse(content={
            "prediction": prediction,
            "confidences": confidences,
            "details": details
        })

    except HTTPException as http_err:
        raise http_err
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Run with Uvicorn locally (optional)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
