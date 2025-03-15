import streamlit as st
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
import os

# Define class names (from your notebook)
class_names = [
    'Blazer', 'Celana_Panjang', 'Celana_Pendek', 'Gaun', 'Hoodie',
    'Jaket', 'Jaket_Denim', 'Jaket_Olahraga', 'Jeans', 'Kaos',
    'Kemeja', 'Mantel', 'Polo', 'Rok', 'Sweter'
]

# Define the model class (ClothesClassifierResNet from your notebook)
class ClothesClassifierResNet(nn.Module):
    def __init__(self, num_classes=len(class_names), dropout_rate=0.5):
        super().__init__()
        self.model = models.resnet50(weights='DEFAULT')
        # Freeze all layers except the final fully connected layer
        for param in self.model.parameters():
            param.requires_grad = False
        # Unfreeze layer4
        for param in self.model.layer4.parameters():
            param.requires_grad = True
        # Replace the final fully connected layer
        self.model.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.model.fc.in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# Prediction function
def predict(image_path, model_path="saved_model.pth"):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    st.write(f"Checking model file at: {os.path.abspath(model_path)}")  # Debug absolute path
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_tensor = image_tensor.to(device)

    # Load the model
    model = ClothesClassifierResNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Perform inference
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted_class = torch.max(output, 1)
        return class_names[predicted_class.item()]

# Streamlit app
st.title("Clothes Image Classifier")
st.write("Upload an image of clothing, and the model will predict its category.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Save the uploaded file temporarily
    temp_file = "temp_image.jpg"
    image.save(temp_file)

    # Predict the class
    try:
        # Model path (assumes saved_model.pth is in the same directory as app.py)
        model_path = "saved_model.pth"
        prediction = predict(temp_file, model_path)
        st.success(f"Predicted Class: **{prediction}**")
    except Exception as e:
        st.error(f"Error: {e}")

    # Clean up the temporary file
    if os.path.exists(temp_file):
        os.remove(temp_file)

# Instructions if model file is missing
if not os.path.exists("saved_model.pth"):
    st.warning(
        "The model file 'saved_model.pth' was not found in the current directory. "
        "Please ensure it is saved in the same directory as this script. "
        "It should have been saved by your Jupyter notebook after training."
    )