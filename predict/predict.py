import torch
import pandas as pd
import os
import sys
from torch.utils.data import DataLoader, Dataset

# Add project root directory to sys.path for model imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import model classes
from models.dkt import DKT
from models.dkt_plus import DKTPlus
from models.dkvmn import DKVMN
from models.sakt import SAKT
from models.gkt import PAM, MHA

def get_model_class(model_name, method=None):
    model_name = model_name.lower()
    if model_name == "dkt":
        return DKT
    elif model_name == "dkt+":
        return DKTPlus
    elif model_name == "dkvmn":
        return DKVMN
    elif model_name == "sakt":
        return SAKT
    elif model_name == "gkt":
        if method == "PAM":
            return PAM
        elif method == "MHA":
            return MHA
        else:
            raise ValueError("Unknown GKT method")
    else:
        raise ValueError(f"Unsupported model: {model_name}")

# Paths
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model_dir = os.path.join(root_dir, "finaltrainedmodels")
data_dir = os.path.join(root_dir, "datasets/quizuploads")

model_filename = "dkt_MyClassroom_20250516084036.pt"
quiz_filename = "quiz.csv"

# Hardcoded skill2idx matching checkpoint (5 skills)
skill2idx = {
    "Addition and Subtraction Integers": 0,
    "Area of Rectangles": 1,
    "Area of Triangle": 2,
    "Comparing Fractions": 3,
    "Converting Fractions to Decimals": 4
}

# Hardcoded student2idx matching training data user ids
student2idx = {
    "80000": 0,
    "80001": 1,
    "80002": 2,
    "80003": 3,
    "80004": 4,
    "80005": 5,
    "80006": 6,
    "80007": 7,
    "80008": 8,
    "80009": 9
}

num_q = 5
emb_size = 100
hidden_size = 200

# Load model
ModelClass = get_model_class(model_filename.split("_")[0].lower())
model_path = os.path.join(model_dir, model_filename)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ModelClass(num_q=num_q, emb_size=emb_size, hidden_size=hidden_size)

state_dict = torch.load(model_path, map_location=device)
state_dict.pop('out_layer.weight', None)
state_dict.pop('out_layer.bias', None)
model.load_state_dict(state_dict, strict=False)

model.to(device)
model.eval()

# Load and filter quiz.csv by hardcoded skills
quiz_df = pd.read_csv(os.path.join(data_dir, quiz_filename))
quiz_df = quiz_df[quiz_df["skill_name"].isin(skill2idx.keys())]

print(f"Filtered quiz_df rows: {len(quiz_df)}")

class QuizDataset(Dataset):
    def __init__(self, df, skill2idx):
        self.skills = [skill2idx[skill] for skill in df["skill_name"]]

    def __len__(self):
        return len(self.skills)

    def __getitem__(self, idx):
        return torch.tensor(self.skills[idx], dtype=torch.long)

quiz_dataset = QuizDataset(quiz_df, skill2idx)
print(f"Quiz dataset length: {len(quiz_dataset)}")

quiz_loader = DataLoader(quiz_dataset, batch_size=32)

# Predict for all students
predictions = {}

with torch.no_grad():
    for student_id, student_idx in student2idx.items():
        user_tensor = torch.tensor([student_idx], dtype=torch.long).to(device)
        user_predictions = []
        for skills in quiz_loader:
            skills = skills.to(device)
            dummy_response = torch.zeros_like(skills, dtype=torch.long).to(device)
            outputs = model(user_tensor.repeat(len(skills)), dummy_response)  # [batch_size, num_q]
            probs = outputs.gather(1, skills.unsqueeze(1)).squeeze(1)  # [batch_size]
            probs = torch.sigmoid(probs).cpu().numpy()
            user_predictions.extend(probs.tolist())
        predictions[student_id] = user_predictions


print(f"Predicted for {len(predictions)} students.")

# Use quiz_dataset length for columns
column_names = [f"Q{i+1}" for i in range(len(quiz_dataset))]

result_df = pd.DataFrame.from_dict(predictions, orient="index", columns=column_names)
result_df.index.name = "user_id"

print("=== Sample Output ===")
print(result_df.head())

result_df.to_csv(os.path.join(data_dir, "new_quiz_predictions.csv"))
print("Predictions saved to new_quiz_predictions.csv")
