import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import torch.nn.functional as F
import numpy as np
import os
import torchvision.transforms as transforms

from dataset import LeukemiaDataset
from utils import load_image_paths
from efficientnet_pytorch import EfficientNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Get the parent directory (project root) for paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Load best model
model_path = os.path.join(parent_dir, "models", "best_model.pth")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")

checkpoint = torch.load(model_path, map_location=device)
model = EfficientNet.from_pretrained("efficientnet-b0", num_classes=2)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

print(f"Model loaded from: {model_path}")

root = os.path.join(parent_dir, "data")
image_paths, labels = load_image_paths(root)

# Use only validation data
from sklearn.model_selection import train_test_split
_, val_paths, _, val_labels = train_test_split(
    image_paths, labels, test_size=0.2, stratify=labels, random_state=42
)

val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

val_dataset = LeukemiaDataset(val_paths, val_labels, val_transform)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

all_labels = []
all_preds = []
all_probs = []

with torch.no_grad():
    for imgs, lbls in val_loader:
        imgs = imgs.to(device)
        lbls = lbls.to(device)
        outputs = model(imgs)
        probs = F.softmax(outputs, dim=1)[:,1]  # cancer prob
        preds = torch.argmax(outputs, dim=1)

        all_labels.extend(lbls.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

# Metrics
print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=["hem","all"]))

print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))

auc = roc_auc_score(all_labels, all_probs)
print("ROC-AUC Score:", auc)
