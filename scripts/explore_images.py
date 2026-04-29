# explore_images.py
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

print("=== Exploring Your Image Data ===")

# Paths to your data
train_path = "D:/ML_PROJECT/data/archive/digits/train_images"
test_path = "D:/ML_PROJECT/data/archive/digits/test_images"

# Check what we have
def explore_folder(folder_path, folder_name):
    print(f"\n--- {folder_name} ---")
    if os.path.exists(folder_path):
        if folder_name == "train_images":
            # For train_images, we have subfolders 0-9
            for digit_folder in sorted(os.listdir(folder_path)):
                digit_path = os.path.join(folder_path, digit_folder)
                if os.path.isdir(digit_path):
                    num_images = len([f for f in os.listdir(digit_path) if f.endswith('.jpg')])
                    print(f"Digit {digit_folder}: {num_images} images")
                    
                    # Show one sample image from each digit
                    if num_images > 0:
                        sample_image = os.listdir(digit_path)[0]
                        img_path = os.path.join(digit_path, sample_image)
                        img = Image.open(img_path)
                        print(f"  Sample image size: {img.size}")
                        
        else:  # test_images
            num_images = len([f for f in os.listdir(folder_path) if f.endswith('.jpg')])
            print(f"Total test images: {num_images}")
            
            if num_images > 0:
                sample_image = os.listdir(folder_path)[0]
                img_path = os.path.join(folder_path, sample_image)
                img = Image.open(img_path)
                print(f"Sample image size: {img.size}")
    else:
        print(f"Folder not found: {folder_path}")

explore_folder(train_path, "train_images")
explore_folder(test_path, "test_images")

# Let's visualize some images
def show_sample_images():
    print("\n--- Showing Sample Images ---")
    plt.figure(figsize=(12, 6))
    
    # Show training samples
    for digit in range(3):  # Show first 3 digits
        digit_path = os.path.join(train_path, str(digit))
        if os.path.exists(digit_path) and os.listdir(digit_path):
            sample_image = os.listdir(digit_path)[0]
            img_path = os.path.join(digit_path, sample_image)
            
            plt.subplot(2, 3, digit + 1)
            img = Image.open(img_path)
            plt.imshow(img, cmap='gray')
            plt.title(f'Training: Digit {digit}')
            plt.axis('off')
    
    # Show test samples
    test_images = [f for f in os.listdir(test_path) if f.endswith('.jpg')]
    for i in range(3):  # Show first 3 test images
        if i < len(test_images):
            img_path = os.path.join(test_path, test_images[i])
            
            plt.subplot(2, 3, i + 4)
            img = Image.open(img_path)
            plt.imshow(img, cmap='gray')
            plt.title(f'Test Image {i+1}')
            plt.axis('off')
    
    plt.tight_layout()
    plt.show()

show_sample_images()