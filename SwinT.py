import pandas as pd
import numpy as np
from PIL import Image
import os
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

DATA_DIR = r'P:/XAIM Biolab/2nd sem CVDL/HAM10000/'

# Load metadata
num_rows_to_read = None  # To load the whole dataset
data = pd.read_csv(os.path.join(DATA_DIR, 'HAM10000_metadata.csv'), nrows=num_rows_to_read)
data['image_path'] = DATA_DIR + 'HAM10000_all_images/' + data['image_id'] + '.jpg'

# Mapping dictionary for better readability
lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}

data['cell_type'] = data['dx'].map(lesion_type_dict.get)
data['cell_type_idx'] = pd.Categorical(data['cell_type']).codes

# Split dataset into train, validation, and test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2 (20% validation set)

# Define the Swin Transformer model architecture
class SwinTransformer(nn.Module):
    def __init__(self, num_classes):
        super(SwinTransformer, self).__init__()
        # Define the layers of the model
        self.conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.linear = nn.Linear(64 * 224 * 224, num_classes)  # Assuming input image size is 224x224
        self.dropout = nn.Dropout(0.5)  # Dropout layer with a dropout probability of 0.5

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten the feature map
        x = self.dropout(x)  # Apply dropout
        x = self.linear(x)
        return x

# Define the training dataset class
class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['image_path']
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(self.data.iloc[idx]['cell_type_idx'], dtype=torch.long)  # Convert label to torch.long
        return img, label

# Define transformations for data augmentation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),  # Random horizontal flip for data augmentation
    transforms.RandomRotation(15),  # Random rotation of up to 15 degrees for data augmentation
    transforms.ToTensor(),
])

# Create datasets and data loaders
train_dataset = CustomDataset(train_data, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = CustomDataset(val_data, transform=transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
]))
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

test_dataset = CustomDataset(test_data, transform=transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
]))
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Instantiate the Swin Transformer model
model = SwinTransformer(num_classes=7)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 20
train_losses = []
val_losses = []
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    train_losses.append(running_loss / len(train_loader.dataset))

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
    val_losses.append(val_loss / len(val_loader.dataset))

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

# Plot training history
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

# Evaluate the model on validation data
correct = 0
total = 0
with torch.no_grad():
    for images, labels in val_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

val_accuracy = correct / total
print(f'Validation Accuracy: {val_accuracy:.4f}')

# Evaluation on test set
model.eval()
test_correct = 0
test_total = 0
all_predictions = []
all_labels = []
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()
        all_predictions.extend(predicted.tolist())
        all_labels.extend(labels.tolist())

test_accuracy = test_correct / test_total
print(f'Test Accuracy: {test_accuracy:.4f}')

# Confusion matrix and classification report
confusion_mat = confusion_matrix(all_labels, all_predictions)
print("Confusion Matrix:")
print(confusion_mat)

print("\nClassification Report:")
print(classification_report(all_labels, all_predictions))

# Visualize some predictions
for i in range(min(5, len(all_labels))):  # Ensure that we don't exceed the length of the lists
    print(f"True Label: {all_labels[i]}, Predicted Label: {all_predictions[i]}")
