import torch
import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import json

# Define the model architecture (same as in api.py)
import torch.nn as nn
from torchvision import models

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

def preprocess_image(image_path):
    """Preprocess an image for model input"""
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def load_model():
    """Load the trained model"""
    print(f"Loading model from {MODEL_PATH}")
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        return None, None
    
    checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    
    # Extract model information
    print(f"Checkpoint type: {type(checkpoint)}")
    if isinstance(checkpoint, dict):
        print(f"Checkpoint keys: {checkpoint.keys()}")
        
        if 'class_names' in checkpoint:
            class_names = checkpoint['class_names']
            print(f"Found class names: {class_names}")
        else:
            class_names = DEFAULT_CLASSES
            print(f"Using default class names: {class_names}")
            
        if 'model_state_dict' in checkpoint:
            model_state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            model_state_dict = checkpoint['state_dict']
        else:
            model_state_dict = checkpoint
            
        # Check if class_mapping exists
        if 'class_mapping' in checkpoint:
            class_mapping = checkpoint['class_mapping']
            print(f"Found class mapping: {class_mapping}")
    else:
        class_names = DEFAULT_CLASSES
        model_state_dict = checkpoint
        print(f"Using default class names: {class_names}")
        
    num_classes = len(class_names) if isinstance(class_names, list) else 7
    
    # Initialize model
    model = SkinLesionModel(num_classes=num_classes)
    
    # Load state dict
    try:
        model.load_state_dict(model_state_dict)
        print("Model state dictionary loaded successfully")
    except Exception as e:
        print(f"Error loading state dictionary: {e}")
        # Try to handle potential key mismatches
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in model_state_dict.items():
            name = k
            if k.startswith('module.'):
                name = k[7:]  # Remove 'module.' prefix
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        print("Model loaded with key remapping")
    
    model.eval()
    return model, class_names

def test_random_input():
    """Test the model with random input"""
    model, class_names = load_model()
    if model is None:
        return
    
    # Create random input
    random_input = torch.rand(1, 3, 300, 300)
    
    # Run inference
    with torch.no_grad():
        outputs = model(random_input)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
    
    # Print results
    print("\nRandom Input Test Results:")
    print(f"Raw output: {outputs}")
    print(f"Probabilities: {probs}")
    
    # Get prediction
    predicted_idx = torch.argmax(probs).item()
    
    # Map to class name
    if isinstance(class_names, list) and len(class_names) > predicted_idx:
        predicted_class = class_names[predicted_idx]
        if predicted_class in DEFAULT_CLASSES:
            print(f"Predicted class: {predicted_class} ({CLASS_DESCRIPTIONS[predicted_class]})")
        else:
            print(f"Predicted class: {predicted_class}")
    else:
        if predicted_idx < len(DEFAULT_CLASSES):
            print(f"Predicted class: {DEFAULT_CLASSES[predicted_idx]} ({CLASS_DESCRIPTIONS[DEFAULT_CLASSES[predicted_idx]]})")
        else:
            print(f"Predicted index: {predicted_idx} (out of range)")
    
    # Print all class probabilities
    print("\nClass probabilities:")
    for i, prob in enumerate(probs):
        class_name = class_names[i] if isinstance(class_names, list) and i < len(class_names) else f"Class {i}"
        print(f"{class_name}: {prob.item():.4f}")

def save_model_info():
    """Save model information to a JSON file"""
    model, class_names = load_model()
    if model is None:
        return
    
    # Get model information
    info = {
        "architecture": "EfficientNet-B3",
        "num_classes": len(class_names) if isinstance(class_names, list) else 7,
        "class_names": class_names if isinstance(class_names, list) else DEFAULT_CLASSES,
        "input_size": [300, 300],
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        }
    }
    
    # Save to file
    with open("model_info.json", "w") as f:
        json.dump(info, f, indent=2)
    
    print(f"Model information saved to model_info.json")

if __name__ == "__main__":
    print("Verifying skin lesion model...")
    test_random_input()
    save_model_info()
    print("\nVerification complete.")
