# Crowd Demographics Analyzer

An advanced AI-powered system for real-time age and gender prediction from images and videos using state-of-the-art deep learning models and digital image processing techniques.

## 🎯 Overview

This project implements a comprehensive crowd demographics analysis system that combines computer vision, deep learning, and digital image processing (DIP) to accurately predict age groups and gender from facial images. The system is designed for both static images and real-time video processing.

## ✨ Features

- **Real-time Face Detection**: YuNet face detector for robust face localization
- **Age Group Classification**: Categorizes individuals into 6 age groups (0-10, 11-20, 21-30, 31-40, 41-50, 50+)
- **Gender Prediction**: Binary classification (Male/Female)
- **Advanced Feature Extraction**: CLIP (Contrastive Language-Image Pre-training) model for rich visual representations
- **Digital Image Processing**: Contrast enhancement, noise reduction, and image preprocessing
- **Multiple Classifier Support**: MLP, Naive Bayes, Random Forest, and AdaBoost
- **Video Processing**: Real-time demographics analysis from video streams
- **Visualization Tools**: t-SNE visualizations, confusion matrices, and SHAP explanations
- **Batch Processing**: Process multiple images from folders efficiently

## 🏗️ Architecture

### Technology Stack

- **Deep Learning Framework**: PyTorch
- **Feature Extraction**: OpenAI CLIP (ViT-Large-Patch14)
- **Face Detection**: YuNet (ONNX)
- **Classification Models**: Scikit-learn (MLP, Random Forest, Naive Bayes, AdaBoost)
- **Image Processing**: OpenCV, PIL, NumPy
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Explainability**: SHAP (SHapley Additive exPlanations)

### System Pipeline

```
Input Image/Video
    ↓
Face Detection (YuNet)
    ↓
Image Preprocessing (DIP)
    ↓
Feature Extraction (CLIP)
    ↓
Classification (MLP/RF/NB/AdaBoost)
    ↓
Age Group & Gender Prediction
```

## 📊 Dataset

**UTKFace Dataset**: A large-scale face dataset with age, gender, and ethnicity annotations
- **Total Images**: 20,000+ (reduced from original for efficiency)
- **Age Range**: 0-116 years
- **Gender Distribution**: Balanced male/female samples
- **Data Split**: 80% training, 20% testing

### Age Categories

| Category | Age Range | Description |
|----------|-----------|-------------|
| 0-10     | 0-10 years | Children |
| 11-20    | 11-20 years | Teenagers |
| 21-30    | 21-30 years | Young Adults |
| 31-40    | 31-40 years | Adults |
| 41-50    | 41-50 years | Middle-aged |
| 50+      | 50+ years | Seniors |

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for faster processing)
- 8GB+ RAM

### Setup

```bash
# Clone the repository
git clone https://github.com/Mohammed0Arfath/Crowd-Demographics-Analyzer.git
cd Crowd-Demographics-Analyzer

# Install dependencies
pip install torch torchvision transformers pillow scikit-learn pandas tqdm matplotlib opencv-python joblib shap plotly seaborn
```

## 🚀 Usage

### 1. Feature Extraction

```python
from transformers import CLIPProcessor, CLIPModel
import torch

# Load CLIP model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# Extract features
features = extract_features(train_df, preprocess=False)
```

### 2. Train Models

```python
# Train Age Prediction Model
mlp_age = train_mlp(X_train, y_train_age, X_test, y_test_age, 
                     task="age", 
                     model_filename="mlp_age_raw.pkl")

# Train Gender Prediction Model
mlp_gender = train_mlp(X_train, y_train_gender, X_test, y_test_gender, 
                        task="gender", 
                        model_filename="mlp_gender_raw.pkl")
```

### 3. Real-time Video Processing

```python
# Process video with demographics analysis
video_path = "input_video.mp4"
output_path = "output_video.mp4"
process_video(video_path, output_path)
```

### 4. Batch Image Processing

```python
# Process all images in a folder
folder_path = "test_images/"
process_images_in_folder(folder_path)
```

## 📈 Model Performance

### Age Prediction

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| MLP (Raw) | 68.5% | 0.67 | 0.68 | 0.67 |
| MLP (Preprocessed) | 71.2% | 0.70 | 0.71 | 0.70 |
| Random Forest | 65.3% | 0.64 | 0.65 | 0.64 |
| Naive Bayes | 58.7% | 0.57 | 0.59 | 0.58 |

### Gender Prediction

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| MLP (Raw) | 92.4% | 0.92 | 0.92 | 0.92 |
| MLP (Preprocessed) | 93.7% | 0.94 | 0.94 | 0.94 |

## 🔬 Digital Image Processing Techniques

### Preprocessing Pipeline

1. **Face Detection**: YuNet ONNX model with adaptive thresholding
2. **Contrast Enhancement**: ImageEnhance with 1.1x contrast boost
3. **Noise Reduction**: Median Filter (kernel size: 3x3)
4. **Normalization**: Pixel values scaled to [0, 1]
5. **Resizing**: LANCZOS resampling to 384x384 or 256x256

### Comparison: Raw vs. Preprocessed

The system supports both raw and preprocessed image analysis, showing:
- **2-5% improvement** in accuracy with preprocessing
- **Better generalization** on diverse lighting conditions
- **Reduced noise** in feature representations

## 📊 Visualizations

### Feature Space Analysis

- **t-SNE Plots**: 2D and 3D visualizations of learned features
- **Interactive 3D Plots**: Plotly-based rotatable visualizations
- **Confusion Matrices**: Heatmap representations of model predictions

### Explainability

- **SHAP Values**: Feature importance analysis using Gradient Explainer
- **Attention Maps**: Visualization of regions contributing to predictions
- **Age-specific Analysis**: Understanding model focus across different age groups

## 📁 Project Structure

```
Crowd-Demographics-Analyzer/
├── Age_and_Gender_Prediction_adv.ipynb   # Main notebook
├── real time.ipynb                        # Real-time processing notebook
├── models/
│   ├── face_detection_yunet_2023mar.onnx # Face detection model
│   ├── mlp_age_raw_adv.pkl               # Age prediction model
│   ├── mlp_gender_raw1_adv.pkl           # Gender prediction model
│   └── haarcascade_frontalface_default.xml
├── test data/                             # Test images
├── UTKFace/                               # Dataset (not included)
└── README.md                              # This file
```

## 🎓 Applications

- **Retail Analytics**: Customer demographics tracking
- **Security Systems**: Age-restricted area monitoring
- **Marketing Research**: Audience analysis for campaigns
- **Smart Cities**: Public space demographic insights
- **Event Management**: Crowd composition analysis
- **Healthcare**: Patient flow demographics

## 🔧 Configuration

### Model Parameters

**MLP Classifier:**
```python
hidden_layer_sizes=(1024, 512, 256, 128)
activation='relu'
solver='adam'
learning_rate_init=0.001
max_iter=500
```

**YuNet Face Detector:**
```python
input_size=(320, 320)
score_threshold=0.9
nms_threshold=0.3
```

## 📊 Performance Optimization

- **GPU Acceleration**: CUDA support for model inference
- **Batch Processing**: Efficient video frame processing
- **Feature Caching**: Joblib-based feature storage
- **Model Compression**: ONNX format for face detection

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open-source and available under the MIT License.

## 👨‍💻 Author

**Mohammed Arfath**
- GitHub: [@Mohammed0Arfath](https://github.com/Mohammed0Arfath)

## 🙏 Acknowledgments

- **UTKFace Dataset** by Zhang et al.
- **OpenAI CLIP** for powerful visual representations
- **YuNet** face detection by OpenCV
- **Scikit-learn** for machine learning tools

## 📚 References

1. Radford, A., et al. (2021). "Learning Transferable Visual Models From Natural Language Supervision." ICML.
2. Zhang, Z., et al. (2017). "Age Progression/Regression by Conditional Adversarial Autoencoder." CVPR.
3. Yu, S., et al. (2023). "YuNet: A Fast and Accurate CNN-based Face Detector." arXiv.

## 📧 Contact

For questions or collaboration opportunities, please open an issue or contact through GitHub.

---

⭐ **Star this repository** if you found it helpful!
