import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from torchvision.models import EfficientNet_B3_Weights
from sklearn.model_selection import train_test_split
from PIL import Image, UnidentifiedImageError
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define paths
DATA_DIR = 'data/HAM10000/'
METADATA_PATH = os.path.join(DATA_DIR, 'HAM10000_metadata.csv')
IMAGE_DIR = os.path.join(DATA_DIR, 'images/')
MODEL_SAVE_PATH = 'skin_lesion_model.pth'

# Load metadata
metadata = pd.read_csv(METADATA_PATH)

# Map diagnosis to numerical labels
diagnosis_mapping = {
    'akiec': 0,
    'bcc': 1,
    'bkl': 2,
    'df': 3,
    'mel': 4,
    'nv': 5,
    'vasc': 6
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
        try:
            image = Image.open(self.images[idx]).convert('RGB')
        except (UnidentifiedImageError, FileNotFoundError):
            # Return a dummy black image if failed to load
            image = Image.new("RGB", (300, 300))
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Prepare file paths
image_paths = [os.path.join(IMAGE_DIR, img_id + '.jpg') for img_id in metadata['image_id']]
labels = metadata['label'].values

# Split data
train_images, test_images, train_labels, test_labels = train_test_split(
    image_paths, labels, test_size=0.2, random_state=42, stratify=labels
)
val_images, test_images, val_labels, test_labels = train_test_split(
    test_images, test_labels, test_size=0.5, random_state=42, stratify=test_labels
)

# Data augmentation and normalization
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Datasets
train_dataset = SkinLesionDataset(train_images, train_labels, train_transforms)
val_dataset = SkinLesionDataset(val_images, val_labels, val_transforms)
test_dataset = SkinLesionDataset(test_images, test_labels, val_transforms)

# Weighted sampling to handle class imbalance
train_labels_tensor = torch.tensor(train_labels)
class_counts = np.bincount(train_labels_tensor.numpy())
class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
class_weights[4] *= 2.0  # Prioritize 'mel' (melanoma)
sample_weights = class_weights[train_labels_tensor]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

# Model definition
class SkinLesionModel(nn.Module):
    def __init__(self, num_classes=7):
        super(SkinLesionModel, self).__init__()
        weights = EfficientNet_B3_Weights.DEFAULT
        self.efficientnet = models.efficientnet_b3(weights=weights)
        in_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(0.3, inplace=True),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.efficientnet(x)

model = SkinLesionModel()

# Loss, optimizer, scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, verbose=True)

# Train and validate functions
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for inputs, labels in tqdm(dataloader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)  # Move inputs and labels to GPU
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validation"):
            inputs, labels = inputs.to(device), labels.to(device)  # Move inputs and labels to GPU
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=30):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Ensure GPU is selected if available
    model.to(device)  # Move model to the selected device
    best_val_acc, counter, patience = 0.0, 0, 5
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print("Model saved.")
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping.")
                break

    return history

# Train model
history = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler)

# Evaluate on test set
model.load_state_dict(torch.load(MODEL_SAVE_PATH))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
test_loss, test_acc = validate(model, test_loader, criterion, device)
print(f"\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

# Plot training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Loss')

plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='Train Acc')
plt.plot(history['val_acc'], label='Val Acc')
plt.legend()
plt.title('Accuracy')

plt.tight_layout()
plt.savefig('training_history.png')
plt.show()

# Confusion matrix
def plot_confusion_matrix(model, dataloader, class_names):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())

    from sklearn.metrics import confusion_matrix, classification_report
    import seaborn as sns
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.show()

    print("\nClassification Report:\n")
    print(classification_report(all_labels, all_preds, target_names=class_names))

class_names = list(diagnosis_mapping.keys())
plot_confusion_matrix(model, test_loader, class_names)
