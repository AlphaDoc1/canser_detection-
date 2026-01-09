import torch
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import numpy as np
import os
import sys

# Add src directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
src_dir = os.path.join(parent_dir, "src")
sys.path.insert(0, src_dir)

from dataset import LeukemiaDataset
from utils import load_image_paths

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Get paths relative to project root
model_path = os.path.join(parent_dir, "models", "best_model.pth")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")

print(f"Loading model from: {model_path}")
checkpoint = torch.load(model_path, map_location=device)
model = EfficientNet.from_pretrained("efficientnet-b0", num_classes=2)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# Prepare data
root = os.path.join(parent_dir, "data")
print(f"Loading data from: {root}")
image_paths, labels = load_image_paths(root)
print(f"Total images: {len(image_paths)}")
_, val_paths, _, val_labels = train_test_split(image_paths, labels, test_size=0.2, stratify=labels, random_state=42)
print(f"Validation images: {len(val_paths)}")

val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])

dataset = LeukemiaDataset(val_paths, val_labels, val_transform)
loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

all_preds = []
all_labels = []
all_probs = []

print("Running predictions...")
with torch.no_grad():
    for imgs, lbls in loader:
        imgs = imgs.to(device)
        lbls = lbls.to(device)
        out = model(imgs)
        probs = F.softmax(out, dim=1)[:,1]  # Probability of class 1 (ALL)
        preds = torch.argmax(out, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(lbls.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

print(f"Predictions complete. Total: {len(all_preds)}")

# Create output directory (relative to proof directory)
output_dir = os.path.join(current_dir, "proof_metrics")
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory: {output_dir}")

# Save classification report
report = classification_report(all_labels, all_preds, target_names=["HEM","ALL"])
report_path = os.path.join(output_dir, "classification_report.txt")
with open(report_path, "w") as f:
    f.write(report)
print(f"✓ Saved classification report to: {report_path}")
print("\nClassification Report:")
print(report)

# Save confusion matrix image
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(cm, display_labels=["HEM","ALL"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix", fontsize=14, fontweight='bold')
plt.tight_layout()
cm_path = os.path.join(output_dir, "confusion_matrix.png")
plt.savefig(cm_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Saved confusion matrix to: {cm_path}")
print("\nConfusion Matrix:")
print(cm)

# ROC curve
fpr, tpr, _ = roc_curve(all_labels, all_probs)
auc = roc_auc_score(all_labels, all_probs)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}", linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.title("ROC Curve", fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
roc_path = os.path.join(output_dir, "roc_auc.png")
plt.savefig(roc_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Saved ROC curve to: {roc_path}")
print(f"\nROC-AUC Score: {auc:.4f}")

# Calculate and display additional metrics
accuracy = (np.array(all_preds) == np.array(all_labels)).mean() * 100
print(f"\nOverall Accuracy: {accuracy:.2f}%")
print(f"\n✅ All metrics exported to: {output_dir}")
