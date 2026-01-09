import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.to(device)
        self.model.eval()
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Use proper closures to capture self
        def save_activation(module, input, output):
            self.activations = output.detach()
        
        def save_gradient(module, grad_input, grad_output):
            if grad_output[0] is not None:
                self.gradients = grad_output[0].detach()
        
        # Register hooks
        self.forward_handle = target_layer.register_forward_hook(save_activation)
        self.backward_handle = target_layer.register_full_backward_hook(save_gradient)

    def generate(self, image_tensor):
        self.model.zero_grad()
        
        output = self.model(image_tensor)
        output_idx = output.argmax()
        score = output[0, output_idx]
        
        score.backward(retain_graph=True)
        
        if self.gradients is None or self.activations is None:
            raise RuntimeError("Gradients or activations not captured. Check hook registration.")
        
        # Calculate CAM
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=False).squeeze()
        cam = torch.relu(cam)
        cam = cam.cpu().numpy()
        
        # Normalize
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)
        
        return cam, output_idx.item()
    
    def __del__(self):
        # Clean up hooks
        if hasattr(self, 'forward_handle'):
            self.forward_handle.remove()
        if hasattr(self, 'backward_handle'):
            self.backward_handle.remove()

def apply_heatmap(image, cam, alpha=0.4):
    """Apply heatmap overlay to image."""
    # Ensure cam is 2D
    if len(cam.shape) == 1:
        h = int(np.sqrt(cam.shape[0]))
        cam = cam.reshape(h, h)
    
    # Resize CAM to match image size
    cam_resized = cv2.resize(cam, (image.shape[1], image.shape[0]))
    
    # Create heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Overlay
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    return overlay

if __name__ == "__main__":
    # Get the parent directory (project root) for paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    
    # Load model
    model_path = os.path.join(parent_dir, "models", "best_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")
    
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    model = EfficientNet.from_pretrained("efficientnet-b0", num_classes=2)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Get target layer for GradCAM (last conv layer in EfficientNet)
    target_layer = model._blocks[-1]._project_conv
    gradcam = GradCAM(model, target_layer)
    
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Get image path
    img_path = input("Enter image path: ").strip().strip('"').strip("'")
    
    # If relative path, try relative to project root first
    if not os.path.isabs(img_path):
        full_path = os.path.join(parent_dir, img_path)
        if os.path.exists(full_path):
            img_path = full_path
    
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")
    
    print(f"Processing image: {img_path}")
    image = Image.open(img_path).convert("RGB")
    orig_size = image.size
    input_tensor = transform(image).unsqueeze(0).to(device)

    cam, pred = gradcam.generate(input_tensor)
    
    # Resize original image for visualization
    orig = np.array(image.resize((224, 224)))
    heatmap = apply_heatmap(orig, cam, alpha=0.5)
    
    # Save heatmap
    output_dir = os.path.join(parent_dir, "outputs", "heatmaps")
    os.makedirs(output_dir, exist_ok=True)
    
    image_name = os.path.splitext(os.path.basename(img_path))[0]
    output_path = os.path.join(output_dir, f"{image_name}_gradcam.png")
    
    # Convert RGB to BGR for cv2.imwrite
    heatmap_bgr = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, heatmap_bgr)
    
    class_names = ["hem", "all"]
    print(f"\nPrediction: {class_names[pred]}")
    print(f"Saved heatmap to: {output_path}")
