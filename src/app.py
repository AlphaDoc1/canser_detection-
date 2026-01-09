import streamlit as st
import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
import cv2
from gradcam import GradCAM, apply_heatmap
import os
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get the parent directory (project root) for paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

@st.cache_resource
def load_model():
    model_path = os.path.join(parent_dir, "models", "best_model.pth")
    
    if not os.path.exists(model_path):
        st.error(f"❌ Model not found at {model_path}")
        st.error("Please train the model first by running: `python src/train.py`")
        st.stop()
    
    checkpoint = torch.load(model_path, map_location=device)
    model = EfficientNet.from_pretrained("efficientnet-b0", num_classes=2)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

try:
    model = load_model()
    target_layer = model._blocks[-1]._project_conv
    gradcam = GradCAM(model, target_layer)
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])

st.title("🩺 Leukemia Detection System")
st.write("Upload a microscopic image to predict cancerous vs non-cancerous and view Grad-CAM heatmap.")

# Sidebar with device info
with st.sidebar:
    st.header("ℹ️ System Info")
    st.write(f"**Device:** {device}")
    if torch.cuda.is_available():
        st.write(f"**GPU:** {torch.cuda.get_device_name(0)}")
    st.write(f"**Model:** EfficientNet-B0")
    
    # Show model accuracy if available
    try:
        model_path = os.path.join(parent_dir, "models", "best_model.pth")
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu')
            if 'val_acc' in checkpoint:
                st.write(f"**Validation Accuracy:** {checkpoint['val_acc']:.2f}%")
    except:
        pass

uploaded_file = st.file_uploader("Upload Image", type=['png','jpg','jpeg','bmp'])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption='Uploaded Image', width=300)

        # Show loading indicator
        with st.spinner("Processing image..."):
            input_tensor = transform(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = F.softmax(outputs, dim=1)[0]
                pred = torch.argmax(probs).item()
                conf = probs[pred].item()

        label = "Cancerous (ALL)" if pred == 1 else "Non-Cancerous (HEM)"
        prob_hem = probs[0].item()
        prob_all = probs[1].item()
        
        # Prediction display
        col1, col2 = st.columns(2)
        with col1:
            st.write("### Prediction")
            if pred == 1:
                st.success(f"**{label}**")
            else:
                st.info(f"**{label}**")
        
        with col2:
            st.write("### Confidence")
            st.metric("Confidence Score", f"{conf*100:.2f}%", label_visibility="hidden")
        
        # Probability breakdown
        st.write("### Probability Breakdown")
        prob_col1, prob_col2 = st.columns(2)
        with prob_col1:
            st.progress(prob_hem, text=f"Non-Cancerous (HEM): {prob_hem*100:.2f}%")
        with prob_col2:
            st.progress(prob_all, text=f"Cancerous (ALL): {prob_all*100:.2f}%")

        # GradCAM
        with st.spinner("Generating Grad-CAM heatmap..."):
            try:
                cam, _ = gradcam.generate(input_tensor)
                orig = np.array(image.resize((224, 224)))
                heatmap = apply_heatmap(orig, cam)

                st.write("### Grad-CAM Heatmap")
                st.image(heatmap, width=300, caption="Visual explanation of model prediction")
            except Exception as e:
                st.warning(f"Could not generate heatmap: {str(e)}")
                st.info("Prediction completed successfully, but heatmap generation failed.")
    
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        st.info("Please make sure you uploaded a valid image file.")
