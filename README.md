# Clothes Image Classifier
![Python Version](https://img.shields.io/badge/python-3.13-blue.svg)  
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)  
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)

A deep learning-based clothes image classifier built with PyTorch and deployed as a web app using Streamlit. The project uses a pre-trained ResNet-50 model fine-tuned on a custom dataset to classify clothing items into 15 categories. The trained model achieves approximately 80% validation accuracy.

## Features
- **Model**: Fine-tuned ResNet-50 for classifying 15 clothing categories.
- **Web App**: Streamlit interface for uploading images and getting predictions.
- **Dataset**: Custom dataset with 15 clothing classes (e.g., Blazer, Jeans, Kaos).
- **Training**: Jupyter notebook with training code and evaluation metrics.

## Clothing Categories
- Blazer
- Celana_Panjang (Long Pants)
- Celana_Pendek (Short Pants)
- Gaun (Dress)
- Hoodie
- Jaket (Jacket)
- Jaket_Denim (Denim Jacket)
- Jaket_Olahraga (Sports Jacket)
- Jeans
- Kaos (T-shirt)
- Kemeja (Shirt)
- Mantel (Coat)
- Polo
- Rok (Skirt)
- Sweter (Sweater)

## Prerequisites
- Python 3.13 (or compatible version)
- Required libraries:
  - `torch`
  - `torchvision`
  - `streamlit`
  - `pillow`
- A trained model file (`saved_model.pth`) from the Jupyter notebook.

clothes-image-classifier/
├── app.py                  # Streamlit app for prediction
├── saved_model.pth         # Trained ResNet-50 model weights
├── clothes_classifier_training.ipynb  # Jupyter notebook for training
├── dataset/                # (Optional) Dataset folder (not included in repo)
│   ├── Blazer/
│   ├── Celana_Panjang/
│   └── ... (other classes)
├── README.md               # This file
└── requirements.txt        # List of Python dependencies

torch>=2.0.0
torchvision>=0.15.0
streamlit>=1.0.0
pillow>=9.0.0

##Training Details
Model: ResNet-50 with frozen layers except layer4 and a custom fully connected layer.
Dataset: 7,500 images (5,625 train, 1,875 validation) across 15 classes.
Training: 10 epochs, Adam optimizer (lr=0.001), CrossEntropyLoss.
Accuracy: ~80.37% on validation set.
See clothes_classifier_training.ipynb for full details.


---

### How to Use:
1. **Copy the Text**:
   - Copy the entire block above.

2. **Create `README.md`**:
   - Open a text editor (e.g., Notepad, VS Code), paste the content, and save as `README.md` in your project directory.

3. **Customize**:
   - Replace `your-username` with your GitHub username.
   - Update the model download link if applicable.

4. **Push to GitHub**:
   - Add it to your repo and push:
     ```bash
     git add README.md
     git commit -m "Add README"
     git push origin main

Kaggle Dataset 

    (https://www.kaggle.com/datasets/ryanbadai/clothes-dataset)

