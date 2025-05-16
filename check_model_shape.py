import torch

# Adjust the path to your model file
model_path = '/Users/syauqimuhammad/Documents/KULIAH/Skripsi/hcnohdKT/knowledge-tracing-collection-pytorch/pretrainedmodels/dkt_ASSIST2009_20250509073010.pt'

# Load the model weights
ckpt = torch.load(model_path, map_location='cpu')

print(f"🔍 Inspecting model: {model_path}")
for k, v in ckpt.items():
    print(f"{k}: {v.shape}")
