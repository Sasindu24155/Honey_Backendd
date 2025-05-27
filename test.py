import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
csv_path = 'bee_data.csv'  # Adjust if needed
df = pd.read_csv(csv_path)

# Encode categorical labels (health status)
label_encoder = LabelEncoder()
df['health_encoded'] = label_encoder.fit_transform(df['health'])
num_classes = len(label_encoder.classes_)

# Image settings
image_size = (128, 128)  # Resize images
image_folder = 'bee_imgs/'  # Folder containing images


# Custom Dataset class
class BeeDataset(Dataset):
    def __init__(self, df, image_folder, transform=None):
        self.df = df
        self.image_folder = image_folder
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_folder, self.df.iloc[idx]['file'])
        image = Image.open(img_path).convert('RGB')
        label = self.df.iloc[idx]['health_encoded']

        if self.transform:
            image = self.transform(image)

        return image, label


# Data transformations
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Split dataset
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
train_dataset = BeeDataset(train_df, image_folder, transform)
val_dataset = BeeDataset(val_df, image_folder, transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# Define CNN Model
class BeeHealthCNN(nn.Module):
    def __init__(self, num_classes):
        super(BeeHealthCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BeeHealthCNN(num_classes).to(device)

# Load trained model
model.load_state_dict(torch.load('bee_health_classifier.pth', map_location=device))
model.eval()


# Function to predict image

def predict_image(image_path, model, transform, label_encoder, device):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    with torch.no_grad():
        output = model(image)
        predicted_class = torch.argmax(output, dim=1).item()

    predicted_label = label_encoder.inverse_transform([predicted_class])[0]
    return predicted_label


# Example usage
test_image_path = "030_204.png"  # Example image path
prediction = predict_image(test_image_path, model, transform, label_encoder, device)
print(f"Predicted Health Condition: {prediction}")