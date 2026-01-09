import torch
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
import os
import sys
from tqdm import tqdm

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
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

dataset = LeukemiaDataset(val_paths, val_labels, val_transform)
loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

# Create output directory (relative to proof directory)
output_dir = os.path.join(current_dir, "proof_predictions")
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory: {output_dir}")

results = []

print("Running predictions...")
with torch.no_grad():
    for idx, (image, label) in enumerate(tqdm(loader, desc="Processing", total=len(val_paths))):
        image = image.to(device)
        path = val_paths[idx]  # Get corresponding path
        
        out = model(image)
        probs = F.softmax(out, dim=1)[0]
        pred = torch.argmax(probs).item()
        conf = probs[pred].item()
        
        # Get probabilities for both classes
        prob_hem = probs[0].item()
        prob_all = probs[1].item()

        results.append({
            "image_filename": os.path.basename(path),
            "image_path": path,
            "true_label": "ALL" if label.item() == 1 else "HEM",
            "predicted_label": "ALL" if pred == 1 else "HEM",
            "confidence": round(conf, 4),
            "prob_HEM": round(prob_hem, 4),
            "prob_ALL": round(prob_all, 4),
            "correct": 1 if pred == label.item() else 0
        })

df = pd.DataFrame(results)
output_path = os.path.join(output_dir, "predictions.csv")
df.to_csv(output_path, index=False)

print(f"\n✓ Saved predictions to: {output_path}")
print(f"Total predictions: {len(results)}")
accuracy = df['correct'].mean() * 100
print(f"Accuracy: {accuracy:.2f}%")
print(f"Correct predictions: {df['correct'].sum()}/{len(results)}")

# Show summary
print("\nPrediction Summary:")
print(df['predicted_label'].value_counts())
print("\nTrue Label Distribution:")
print(df['true_label'].value_counts())
