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


csv_path = 'bee_data.csv'
df = pd.read_csv(csv_path)
label_encoder = LabelEncoder()
df['health_encoded'] = label_encoder.fit_transform(df['health'])


num_classes = len(label_encoder.classes_)

image_size = (128, 128)
image_folder = 'bee_imgs/'
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
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
train_dataset = BeeDataset(train_df, image_folder, transform)
val_dataset = BeeDataset(val_df, image_folder, transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BeeHealthCNN(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 10
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
        if batch_idx % 10 == 0:
            print(f'Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}')
    train_acc = correct / total
    print(f'Epoch {epoch + 1}/{epochs}, Loss: {train_loss / len(train_loader):.4f}, Accuracy: {train_acc:.4f}')
torch.save(model.state_dict(), 'bee_health_classifier.pth')