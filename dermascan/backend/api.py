import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image, UnidentifiedImageError
import os
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from starlette.responses import JSONResponse
from io import BytesIO
import torch.nn as nn

# Create FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://derma-scan-kappa.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check
@app.get("/")
async def root():
    return {"status": "ok", "message": "DermaScan API is running. Use /api/predict endpoint for predictions."}

MODEL_PATH = "skin_lesion_model.pth"

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.zeros((1, 7))  # default fallback

# Load model
def load_model():
    try:
        model_path = os.path.join(os.path.dirname(__file__), MODEL_PATH)
        if not os.path.exists(model_path):
            print("❌ Model not found. Using dummy.")
            return DummyModel(), ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        class_names = checkpoint.get('class_names', ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'])
        num_classes = len(class_names)

        model = models.efficientnet_b3(weights=None)
        num_ftrs = model.classifier[1].in_features
        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(num_ftrs, num_classes)
        )

        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        print(f"✅ Loaded trained model with {num_classes} classes: {class_names}")
        return model, class_names

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return DummyModel(), ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

# Image preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Quality analysis helpers
def analyze_image_quality(image):
    width, height = image.size
    if width < 100 or height < 100:
        return "Poor - Image is too small"
    img_array = np.array(image.convert('L'))
    variance = np.var(img_array)
    if variance < 100:
        return "Poor - Image may be blurry"
    elif variance < 500:
        return "Moderate"
    else:
        return "Good"

def analyze_lesion_border(image):
    img_array = np.array(image)
    edge_strength = np.std(img_array)
    if edge_strength < 30:
        return "Poorly-defined"
    elif edge_strength < 60:
        return "Moderately-defined"
    else:
        return "Well-defined"

def analyze_color_variation(image):
    img_rgb = image.convert('RGB')
    img_array = np.array(img_rgb)
    r_std = np.std(img_array[:, :, 0])
    g_std = np.std(img_array[:, :, 1])
    b_std = np.std(img_array[:, :, 2])
    avg_std = (r_std + g_std + b_std) / 3
    if avg_std < 20:
        return "Minimal"
    elif avg_std < 40:
        return "Moderate"
    else:
        return "Significant"

# Prediction endpoint
@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        try:
            image = Image.open(BytesIO(contents)).convert('RGB')
        except UnidentifiedImageError:
            raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")

        model, class_names = load_model()
        processed_image = preprocess_image(image)

        with torch.no_grad():
            outputs = model(processed_image)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]

        confidences = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
        prediction = class_names[torch.argmax(probabilities).item()]

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

    except HTTPException as http_err:
        raise http_err
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Run the app
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
