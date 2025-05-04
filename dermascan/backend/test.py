import torch
import os

# Path to your model file
model_path = os.path.join(os.path.dirname(__file__), 'skin_lesion_model.pth')

# Load the checkpoint
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

# Print the keys in the checkpoint dictionary
print("Checkpoint keys:", checkpoint.keys())
