"use client"

import { useState } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import GradientBackground from "@/components/gradient-background"
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter"
import { tomorrow } from "react-syntax-highlighter/dist/esm/styles/prism"

export default function ModelPage() {
  const [activeTab, setActiveTab] = useState("architecture")

  return (
    <main className="relative min-h-screen overflow-hidden">
      <GradientBackground />

      <div className="container mx-auto px-4 py-12 relative z-10">
        <h1 className="text-3xl md:text-4xl font-bold text-slate-900 dark:text-white mb-8 text-center">
          Model Implementation
        </h1>

        <div className="max-w-5xl mx-auto">
          <Tabs defaultValue="architecture" onValueChange={setActiveTab}>
            <TabsList className="grid w-full grid-cols-4 mb-8">
              <TabsTrigger value="architecture">Architecture</TabsTrigger>
              <TabsTrigger value="training">Training</TabsTrigger>
              <TabsTrigger value="preprocessing">Preprocessing</TabsTrigger>
              <TabsTrigger value="deployment">Deployment</TabsTrigger>
            </TabsList>

            <Card className="bg-white/90 dark:bg-slate-800/90 backdrop-blur-md border-slate-200 dark:border-slate-700">
              <CardContent className="p-6">
                <TabsContent value="architecture" className="mt-0">
                  <h2 className="text-2xl font-semibold text-slate-900 dark:text-white mb-4">Model Architecture</h2>
                  <p className="text-slate-700 dark:text-slate-300 mb-6">
                    We use the EfficientNet-B3 architecture, which provides an excellent balance between accuracy and
                    computational efficiency for skin lesion classification.
                  </p>

                  <SyntaxHighlighter language="python" style={tomorrow} className="rounded-md">
                    {`import torch
import torch.nn as nn
import torchvision.models as models

class SkinLesionModel(nn.Module):
    def __init__(self, num_classes=7):
        super(SkinLesionModel, self).__init__()
        # Load pre-trained EfficientNet-B3
        self.efficientnet = models.efficientnet_b3(pretrained=True)
        
        # Replace the classifier
        in_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, num_classes)
        )
        
    def forward(self, x):
        return self.efficientnet(x)

# Initialize the model
model = SkinLesionModel(num_classes=7)
print(model)`}
                  </SyntaxHighlighter>
                </TabsContent>

                <TabsContent value="training" className="mt-0">
                  <h2 className="text-2xl font-semibold text-slate-900 dark:text-white mb-4">Training Process</h2>
                  <p className="text-slate-700 dark:text-slate-300 mb-6">
                    Our training process includes data augmentation, class balancing, and a learning rate scheduler to
                    handle the imbalanced HAM10000 dataset.
                  </p>

                  <SyntaxHighlighter language="python" style={tomorrow} className="rounded-md">
                    {`import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split

# Data augmentation
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Dataset preparation
train_dataset = SkinLesionDataset(
    images=train_images,
    labels=train_labels,
    transform=train_transforms
)

# Handle class imbalance with weighted sampling
class_counts = [sum(train_labels == i) for i in range(7)]
weights = 1. / torch.tensor(class_counts, dtype=torch.float)
sample_weights = weights[train_labels]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    sampler=sampler,
    num_workers=4
)

# Model, loss function, and optimizer
model = SkinLesionModel(num_classes=7)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3, verbose=True
)

# Training loop
num_epochs = 30
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
    
    # Validation and scheduler step
    val_loss = validate(model, val_loader, criterion, device)
    scheduler.step(val_loss)

# Save the model
torch.save(model.state_dict(), 'skin_lesion_model.pth')`}
                  </SyntaxHighlighter>
                </TabsContent>

                <TabsContent value="preprocessing" className="mt-0">
                  <h2 className="text-2xl font-semibold text-slate-900 dark:text-white mb-4">Data Preprocessing</h2>
                  <p className="text-slate-700 dark:text-slate-300 mb-6">
                    Proper preprocessing of dermatoscopic images is crucial for model performance. Here's how we prepare
                    the HAM10000 dataset.
                  </p>

                  <SyntaxHighlighter language="python" style={tomorrow} className="rounded-md">
                    {`import pandas as pd
import numpy as np
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

# Load metadata
metadata = pd.read_csv('HAM10000_metadata.csv')

# Map diagnosis to numerical labels
diagnosis_mapping = {
    'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3, 
    'mel': 4, 'nv': 5, 'vasc': 6
}
metadata['label'] = metadata['dx'].map(diagnosis_mapping)

# Custom dataset class
class SkinLesionDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# Prepare file paths
image_dir = 'HAM10000_images/'
image_paths = [os.path.join(image_dir, img_id + '.jpg') for img_id in metadata['image_id']]
labels = metadata['label'].values

# Split data
train_images, test_images, train_labels, test_labels = train_test_split(
    image_paths, labels, test_size=0.2, random_state=42, stratify=labels
)

# Further split test into validation and test
val_images, test_images, val_labels, test_labels = train_test_split(
    test_images, test_labels, test_size=0.5, random_state=42, stratify=test_labels
)

print(f"Training samples: {len(train_images)}")
print(f"Validation samples: {len(val_images)}")
print(f"Testing samples: {len(test_images)}")

# Class distribution
for i, name in enumerate(['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']):
    print(f"{name}: {sum(train_labels == i)} training, {sum(val_labels == i)} validation, {sum(test_labels == i)} test")`}
                  </SyntaxHighlighter>
                </TabsContent>

                <TabsContent value="deployment" className="mt-0">
                  <h2 className="text-2xl font-semibold text-slate-900 dark:text-white mb-4">Model Deployment</h2>
                  <p className="text-slate-700 dark:text-slate-300 mb-6">
                    We deploy our trained model using Streamlit, which provides an interactive web interface for skin
                    lesion classification.
                  </p>

                  <SyntaxHighlighter language="python" style={tomorrow} className="rounded-md">
                    {`import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import numpy as np

# Load the model
@st.cache_resource
def load_model():
    model = SkinLesionModel(num_classes=7)
    model.load_state_dict(torch.load('skin_lesion_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

# Preprocess image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Class names
class_names = ['Actinic Keratosis', 'Basal Cell Carcinoma', 'Benign Keratosis',
               'Dermatofibroma', 'Melanoma', 'Melanocytic Nevus', 'Vascular Lesion']

# Main app
def main():
    st.title('Skin Lesion Classification')
    
    # Load model
    model = load_model()
    
    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Predict
        if st.button('Classify'):
            with st.spinner('Analyzing...'):
                # Preprocess
                input_tensor = preprocess_image(image)
                
                # Predict
                with torch.no_grad():
                    output = model(input_tensor)
                    probabilities = torch.nn.functional.softmax(output, dim=1)[0]
                    predicted_class = torch.argmax(probabilities).item()
                
                # Display results
                st.subheader(f'Prediction: {class_names[predicted_class]}')
                
                # Show probabilities
                st.subheader('Confidence Scores:')
                for i, (prob, class_name) in enumerate(zip(probabilities, class_names)):
                    st.progress(float(prob))
                    st.write(f"{class_name}: {prob*100:.2f}%")

if __name__ == '__main__':
    main()`}
                  </SyntaxHighlighter>
                </TabsContent>
              </CardContent>
            </Card>
          </Tabs>
        </div>
      </div>
    </main>
  )
}
