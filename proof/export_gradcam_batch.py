import os
import sys
import cv2
import numpy as np
import torch
from efficientnet_pytorch import EfficientNet
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

# Add src directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
src_dir = os.path.join(parent_dir, "src")
sys.path.insert(0, src_dir)

from gradcam import GradCAM, apply_heatmap

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

target_layer = model._blocks[-1]._project_conv
gradcam = GradCAM(model, target_layer)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

input_dir = os.path.join(parent_dir, "data")
output_dir = os.path.join(current_dir, "proof_heatmaps")
os.makedirs(output_dir, exist_ok=True)

print(f"Input directory: {input_dir}")
print(f"Output directory: {output_dir}")

# Collect all image files first
image_files = []
for root, dirs, files in os.walk(input_dir):
    for file in files:
        if file.lower().endswith((".bmp", ".png", ".jpg", ".jpeg")):
            path = os.path.join(root, file)
            image_files.append(path)

print(f"Found {len(image_files)} images")
print(f"Processing first 10 images...\n")

count = 0
max_images = 10

for path in tqdm(image_files[:max_images], desc="Generating heatmaps"):
    try:
        # Load and process image
        image = Image.open(path).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)
        
        # Generate GradCAM
        cam, pred = gradcam.generate(tensor)
        
        # Resize original image for visualization
        orig = image.resize((224, 224))
        orig_array = np.array(orig)
        heatmap = apply_heatmap(orig_array, cam)
        
        # Create output filename
        image_name = os.path.splitext(os.path.basename(path))[0]
        save_path = os.path.join(output_dir, f"{image_name}_gradcam.png")
        
        # Convert RGB to BGR for cv2.imwrite
        heatmap_bgr = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, heatmap_bgr)
        
        count += 1
        
    except Exception as e:
        print(f"Error processing {path}: {str(e)}")
        continue

print(f"\n✅ Successfully saved {count} Grad-CAM heatmaps to: {output_dir}")
print(f"   Output format: <filename>_gradcam.png")
