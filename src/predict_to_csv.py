import torch
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F
import pandas as pd
import torchvision.transforms as transforms
import os
from tqdm import tqdm

from dataset import LeukemiaDataset
from utils import load_image_paths
from sklearn.model_selection import train_test_split

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

_, val_paths, _, val_labels = train_test_split(
    image_paths, labels, test_size=0.2, stratify=labels, random_state=42
)

val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

dataset = LeukemiaDataset(val_paths, val_labels, val_transform)
loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

results = []

print(f"Processing {len(val_paths)} images...")
with torch.no_grad():
    for idx, (image, label) in enumerate(tqdm(loader, desc="Predicting")):
        image = image.to(device)
        path = val_paths[idx]  # Get corresponding path
        
        outputs = model(image)
        probs = F.softmax(outputs, dim=1)[0]
        pred = torch.argmax(probs).item()
        conf = probs[pred].item()
        
        results.append({
            "image": os.path.basename(path),  # Just filename
            "image_path": path,  # Full path
            "true_label": "all" if label.item() == 1 else "hem",
            "prediction": "all" if pred == 1 else "hem",
            "confidence": round(conf, 4),
            "prob_all": round(probs[1].item(), 4),
            "prob_hem": round(probs[0].item(), 4),
            "correct": 1 if pred == label.item() else 0
        })

df = pd.DataFrame(results)
output_dir = os.path.join(parent_dir, "outputs", "csv")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "predictions.csv")
df.to_csv(output_path, index=False)
print(f"\nCSV saved to: {output_path}")
print(f"Total predictions: {len(results)}")
print(f"Accuracy: {df['correct'].mean()*100:.2f}%")

