# 🩺 Leukemia Detection System

A deep learning-based system for detecting leukemia (cancerous vs non-cancerous cells) in microscopic blood cell images using EfficientNet-B0 and Grad-CAM visualization.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.9.1-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements a deep learning solution for automated leukemia detection from microscopic blood cell images. The system uses EfficientNet-B0, a state-of-the-art convolutional neural network, to classify cells as either:
- **ALL (Acute Lymphoblastic Leukemia)** - Cancerous
- **HEM (Healthy)** - Non-cancerous

The system includes:
- Model training pipeline
- Evaluation and metrics generation
- Grad-CAM visualization for interpretability
- Streamlit web application for easy inference
- Batch prediction and export capabilities

## ✨ Features

- **Deep Learning Model**: EfficientNet-B0 architecture with transfer learning
- **Grad-CAM Visualization**: Visual explanation of model predictions
- **Web Application**: Streamlit-based user interface
- **Comprehensive Metrics**: Classification reports, confusion matrices, ROC curves
- **Batch Processing**: Process multiple images at once
- **GPU Support**: CUDA-enabled for faster training and inference
- **Export Capabilities**: CSV exports for predictions and metrics

## 📁 Project Structure

```
arfa mam/
│
├── models/               # Saved trained models
│   └── best_model.pth
│
├── outputs/
│   ├── csv/              # Prediction CSVs
│   └── heatmaps/         # GradCAM heatmaps
│
├── app/                  # Streamlit app directory
│
├── src/
│   ├── dataset.py        # Dataset loading utilities
│   ├── train.py          # Training script
│   ├── evaluate.py       # Model evaluation
│   ├── test.py           # Testing utilities
│   ├── gradcam.py        # Grad-CAM implementation
│   ├── predict_to_csv.py # Batch prediction export
│   ├── utils.py          # Helper functions
│   └── app.py            # Streamlit web app
│
├── proof/
│   ├── export_metrics.py      # Export evaluation metrics
│   ├── export_predictions.py  # Export predictions
│   └── export_gradcam_batch.py # Batch Grad-CAM generation
│
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 🚀 Installation

### Prerequisites

- Python 3.10 or higher
- CUDA-capable GPU (optional, for faster training)
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/AlphaDoc1/canser_detection-.git
cd canser_detection-
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: For GPU support, install CUDA-enabled PyTorch:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 💻 Usage

### 1. Training the Model

```bash
cd src
python train.py
```

**Arguments** (modify in `train.py`):
- `--train_dir`: Training data directory
- `--val_dir`: Validation data directory
- `--batch_size`: Batch size (default: 16)
- `--num_epochs`: Number of epochs (default: 50)
- `--learning_rate`: Learning rate (default: 0.001)

### 2. Evaluating the Model

```bash
cd src
python evaluate.py
```

This will:
- Load the trained model
- Evaluate on validation set
- Print classification report and confusion matrix
- Calculate ROC-AUC score

### 3. Generating Predictions

```bash
cd src
python predict_to_csv.py
```

Outputs predictions to `outputs/csv/predictions.csv`

### 4. Generating Grad-CAM Heatmaps

**Single Image:**
```bash
cd src
python gradcam.py
# Enter image path when prompted
```

**Batch Processing:**
```bash
cd proof
python export_gradcam_batch.py
```

### 5. Exporting Metrics

```bash
cd proof
python export_metrics.py
```

Generates:
- Classification report (`proof_metrics/classification_report.txt`)
- Confusion matrix (`proof_metrics/confusion_matrix.png`)
- ROC curve (`proof_metrics/roc_auc.png`)

### 6. Running the Web Application

```bash
cd src
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

## 🏗️ Model Architecture

- **Base Model**: EfficientNet-B0 (pretrained on ImageNet)
- **Input Size**: 224x224 RGB images
- **Output**: 2 classes (HEM, ALL)
- **Optimizer**: Adam
- **Loss Function**: Cross-Entropy
- **Learning Rate**: 0.001 with ReduceLROnPlateau scheduler
- **Data Augmentation**: Random horizontal flip, rotation, color jitter

## 📊 Results

The model achieves high accuracy on the validation set. Detailed metrics are available in:
- `proof/proof_metrics/classification_report.txt`
- `proof/proof_metrics/confusion_matrix.png`
- `proof/proof_metrics/roc_auc.png`

### Key Metrics:
- **Accuracy**: See evaluation results
- **ROC-AUC**: See ROC curve plot
- **Precision/Recall**: See classification report

## 🔧 Configuration

### Data Format

Organize your data as follows:
```
data/
├── fold_0/
│   └── fold_0/
│       ├── all/      # ALL (cancerous) images
│       └── hem/      # HEM (healthy) images
├── fold_1/
│   └── fold_1/
│       ├── all/
│       └── hem/
└── fold_2/
    └── fold_2/
        ├── all/
        └── hem/
```

### Model Checkpoints

Trained models are saved in `models/`:
- `best_model.pth`: Best model based on validation accuracy

## 🛠️ Development

### Adding New Features

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

### Code Style

- Follow PEP 8 guidelines
- Use meaningful variable names
- Add docstrings to functions
- Comment complex logic

## 📝 Scripts Overview

| Script | Purpose |
|--------|---------|
| `src/train.py` | Train the model |
| `src/evaluate.py` | Evaluate model performance |
| `src/predict_to_csv.py` | Export predictions to CSV |
| `src/gradcam.py` | Generate Grad-CAM visualizations |
| `src/app.py` | Streamlit web application |
| `proof/export_metrics.py` | Export evaluation metrics |
| `proof/export_predictions.py` | Export predictions with details |
| `proof/export_gradcam_batch.py` | Batch Grad-CAM generation |

## 🐛 Troubleshooting

### Common Issues

1. **CUDA out of memory**
   - Reduce batch size in training script
   - Use CPU instead of GPU

2. **Import errors**
   - Ensure virtual environment is activated
   - Install all requirements: `pip install -r requirements.txt`

3. **Model not found**
   - Train the model first: `python src/train.py`
   - Check model path in scripts

4. **Windows multiprocessing errors**
   - Scripts use `num_workers=0` for Windows compatibility

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- **AlphaDoc1** - [GitHub Profile](https://github.com/AlphaDoc1)

## 🙏 Acknowledgments

- EfficientNet architecture by Google Research
- PyTorch team for the deep learning framework
- Streamlit for the web application framework
- All contributors and researchers in medical imaging AI

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**Note**: This system is for research and educational purposes. Always consult medical professionals for actual medical diagnosis.

