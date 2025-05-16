import torch
import os

# Adjust this path to your actual model checkpoint
model_path = os.path.abspath("/Users/syauqimuhammad/Documents/KULIAH/Skripsi/hcnohdKT/knowledge-tracing-collection-pytorch/finaltrainedmodels/dkt_MyClassroom_20250516084036.pt")

# Load checkpoint
checkpoint = torch.load(model_path, map_location='cpu')

print("Checkpoint parameters and their shapes:")
for key, value in checkpoint.items():
    print(f"{key}: {tuple(value.shape)}")
