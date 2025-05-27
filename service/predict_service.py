import os
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from io import BytesIO

# Load labels from bee_data.csv
csv_path = 'bee_data.csv'
df = pd.read_csv(csv_path)
label_encoder = LabelEncoder()
df['health_encoded'] = label_encoder.fit_transform(df['health'])
num_classes = len(label_encoder.classes_)

# Image processing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Define model
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

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BeeHealthCNN(num_classes).to(device)
model.load_state_dict(torch.load('bee_health_classifier.pth', map_location=device))
model.eval()

# API prediction function
def predict_image_api(uploaded_file):
    image = Image.open(BytesIO(uploaded_file.read())).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        predicted_class = torch.argmax(output, dim=1).item()

    predicted_label = label_encoder.inverse_transform([predicted_class])[0]
    return predicted_label
def predict_bee_health(file):
    image = Image.open(file).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        predicted_index = torch.argmax(output, dim=1).item()

    return label_encoder.inverse_transform([predicted_index])[0]
