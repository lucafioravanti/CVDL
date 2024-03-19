# Import libraries
import os
import pandas as pd
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import tensorflow as tf
from vit_keras import vit
import tensorflow_addons as tfa
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import math
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report


DATA_DIR = r'P:/XAIM Biolab/2nd sem CVDL/HAM10000/'

# Load metadata
num_rows_to_read =  None
data = pd.read_csv(os.path.join(DATA_DIR, 'HAM10000_metadata.csv'), nrows=num_rows_to_read)
data['image_path'] = DATA_DIR + 'HAM10000_all_images/' + data['image_id'] + '.jpg'

# Dictionary for lesion types
lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}

# Add columns for better readability
data['cell_type'] = data['dx'].map(lesion_type_dict.get)
data['cell_type_idx'] = pd.Categorical(data['cell_type']).codes

# Add image to the dataset
data['image'] = data['image_path'].apply(lambda x: Image.open(x).convert('RGB'))

# Fill null ages with the mean
data['age'].fillna((data['age'].mean()), inplace=True)

# Function to preprocess images
def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))  # Resize the image to the required dimensions
    img = np.array(img) / 255.0
    return img


# Split dataset into training, validation, and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

# Separate images and labels for training, validation, and testing
train_images = np.array([preprocess_image(img_path) for img_path in train_data['image_path']])
val_images = np.array([preprocess_image(img_path) for img_path in val_data['image_path']])
test_images = np.array([preprocess_image(img_path) for img_path in test_data['image_path']])

train_labels = np.array(train_data['cell_type_idx'])
val_labels = np.array(val_data['cell_type_idx'])
test_labels = np.array(test_data['cell_type_idx'])

# Initialize ViT model
vit_model = vit.vit_b32(
        image_size=224,
        activation='softmax',
        pretrained=True,
        include_top=False,
        pretrained_top=False,
        classes=7
)

# Define model architecture
model = tf.keras.Sequential([
        vit_model,
        tf.keras.layers.Flatten(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(11, activation=tfa.activations.gelu),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(7, 'softmax')
    ],
    name='vision_transformer')

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('vit_model.h5', save_best_only=True)

def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.1
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(step_decay, verbose=1)
callbacks = [early_stopping, model_checkpoint, lr_scheduler]

# Train the model
history = model.fit(
    train_images, train_labels,
    validation_data=(val_images, val_labels),
    epochs=20,
    batch_size=32,
    callbacks=callbacks
)

# Evaluate the model on validation data
val_loss, val_accuracy = model.evaluate(val_images, val_labels)
print(f'Validation Loss: {val_loss:.4f}')
print(f'Validation Accuracy: {val_accuracy:.4f}')

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')

# Plot training history
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.show()


# Make predictions
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)

# Create confusion matrix
confusion_mat = confusion_matrix(test_labels, predicted_labels)

# Display confusion matrix
print("Confusion Matrix:")
print(confusion_mat)

# Classification Report
print("\nClassification Report:")
print(classification_report(test_labels, predicted_labels))

# Visualize some predictions
for i in range(5):
    print(f"True Label: {test_labels[i]}, Predicted Label: {predicted_labels[i]}")
