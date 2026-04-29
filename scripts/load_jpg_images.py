# load_jpg_images.py
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random

print("=== Loading JPG Images ===")

# Correct paths based on your structure
train_path = "D:/ML_PROJECT/data/archive/digits/train_images"
test_path = "D:/ML_PROJECT/data/archive/digits/test_images"

def count_images(folder_path):
    """Count how many JPG images are in each folder"""
    if not os.path.exists(folder_path):
        return 0
    
    count = 0
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.jpg'):
                count += 1
    return count

def load_images_from_folder(folder_path, label=None):
    """Load images and their labels from a folder"""
    images = []
    labels = []
    
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return images, labels
    
    if label is None:
        # For test_images (no subfolders)
        for file in os.listdir(folder_path):
            if file.lower().endswith('.jpg'):
                img_path = os.path.join(folder_path, file)
                try:
                    img = Image.open(img_path).convert('L')  # Convert to grayscale
                    img_array = np.array(img)
                    images.append(img_array)
                    labels.append(-1)  # -1 means unknown label for test images
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
    else:
        # For train_images subfolders (0, 1, 2, etc.)
        for file in os.listdir(folder_path):
            if file.lower().endswith('.jpg'):
                img_path = os.path.join(folder_path, file)
                try:
                    img = Image.open(img_path).convert('L')  # Convert to grayscale
                    img_array = np.array(img)
                    images.append(img_array)
                    labels.append(label)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
    
    return images, labels

print("Counting images...")
# Count training images
total_train = 0
for digit in range(10):
    digit_path = os.path.join(train_path, str(digit))
    count = count_images(digit_path)
    total_train += count
    print(f"Digit {digit}: {count} images")

# Count test images
total_test = count_images(test_path)
print(f"Test images: {total_test}")
print(f"Total images: {total_train + total_test}")

# Load some sample images to verify
print("\nLoading sample images...")

# Load training samples
train_images = []
train_labels = []

for digit in range(3):  # Load first 3 digits for testing
    digit_path = os.path.join(train_path, str(digit))
    images, labels = load_images_from_folder(digit_path, label=digit)
    if images:
        train_images.extend(images)
        train_labels.extend(labels)
        print(f"Loaded {len(images)} images for digit {digit}")

# Load test samples
test_images, test_labels = load_images_from_folder(test_path)
if test_images:
    print(f"Loaded {len(test_images)} test images")

# Convert to numpy arrays for easier handling
if train_images:
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    print(f"Training data shape: {train_images.shape}")
    print(f"Training labels shape: {train_labels.shape}")

if test_images:
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)
    print(f"Test data shape: {test_images.shape}")

# Visualize samples
def visualize_samples(images, labels, title):
    """Display sample images"""
    if len(images) == 0:
        print(f"No images to display for {title}")
        return
    
    plt.figure(figsize=(12, 4))
    num_samples = min(5, len(images))
    
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(images[i], cmap='gray')
        if labels[i] != -1:
            plt.title(f'Label: {labels[i]}')
        else:
            plt.title('Test Image')
        plt.axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# Show samples
if len(train_images) > 0:
    visualize_samples(train_images, train_labels, "Training Samples")

if len(test_images) > 0:
    visualize_samples(test_images, test_labels, "Test Samples")

print("\n=== Next Steps ===")
if total_train > 0 and total_test > 0:
    print("✓ Great! Images loaded successfully.")
    print("Next: We'll create a complete data loader for all images.")
else:
    print("⚠️  No images found. Let's check the file paths and formats.")