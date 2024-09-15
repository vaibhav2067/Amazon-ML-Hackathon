import os
import pandas as pd
import requests
from tqdm import tqdm
import cv2
from PIL import Image
import pytesseract
import torch
from torchvision import models, transforms
import torch.nn as nn
import re

# Set up directories and constants
image_folder = 'garbage'
os.makedirs(image_folder, exist_ok=True)
max_images = 1000

# Load the dataset
train_df = pd.read_csv('student_resource 3/dataset/train.csv')
test_df = pd.read_csv('student_resource 3/dataset/test.csv')

train_df['index'] = range(len(train_df))

# Step 1: Download Images
def download_image(image_url, save_path):
    try:
        response = requests.get(image_url, stream=True)
        if response.status_code == 200:
            with open(save_path, 'wb') as file:
                for chunk in response.iter_content(1024):
                    file.write(chunk)
        else:
            print(f"Failed to download image {image_url}")
    except Exception as e:
        print(f"Error downloading {image_url}: {e}")

# Download images from the training dataset, limit to first 1000
for idx, row in tqdm(train_df.head(max_images).iterrows(), total=min(len(train_df), max_images)):
    image_url = row['image_link']
    image_path = os.path.join(image_folder, f"{row['index']}.jpg")
    download_image(image_url, image_path)

# Step 2: Preprocess Images
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding
    thresh_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY_INV)[1]
    
    # Resize image
    resized_image = cv2.resize(thresh_image, (224, 224))
    
    return resized_image

# Step 3: Feature Extraction Using OCR
def extract_text_from_image(image_path):
    preprocessed_image = preprocess_image(image_path)
    extracted_text = pytesseract.image_to_string(preprocessed_image)
    return extracted_text

print('over')

# Step 4: Feature Extraction Using Deep Learning
# Load pre-trained ResNet model
resnet = models.resnet50(pretrained=True)
resnet.eval()

# Define image transformation
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features(image_path):
    image = Image.open(image_path)
    image_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        features = resnet(image_tensor)
    return features

# Step 5: Predict Entity Values (Placeholder)
def format_entity_value(ocr_text, entity_name):
    match = re.search(r"(\d+\.?\d*)\s*(\w+)", ocr_text)
    if match:
        number, unit = match.groups()
        return f"{float(number)} {unit.lower()}"
    return ""

# Example prediction function (dummy implementation)
def predict_entity_value(image_path, entity_name):
    if entity_name in ["item_weight", "item_volume"]:  # Example entities
        ocr_text = extract_text_from_image(image_path)
        return format_entity_value(ocr_text, entity_name)
    else:
        features = extract_features(image_path)
        # Implement regression model here to predict numeric values (not shown)
        return "Prediction based on features"

# Step 6: Generate Predictions for Test Data
predictions = []
for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
    image_url = row['image_link']
    image_path = os.path.join(image_folder, f"{row['index']}.jpg")
    entity_name = row['entity_name']
    if os.path.exists(image_path):
        prediction = predict_entity_value(image_path, entity_name)
    else:
        prediction = ""
    predictions.append({'index': row['index'], 'prediction': prediction})

# Save predictions to CSV
output_df = pd.DataFrame(predictions)
output_df.to_csv('output.csv', index=False)

print("Predictions saved to output.csv")

# Note: Replace dummy prediction logic with actual implementation
