# optimized_model.py
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import tensorflow as tf
from tensorflow.keras import layers, models
import cv2

def enhanced_preprocess(image, input_type="raw"):
    """
    Enhanced preprocessing for real-world images
    """
    # Convert to grayscale
    if image.mode != 'L':
        image = image.convert('L')
    
    # Convert to numpy for OpenCV processing
    img_array = np.array(image)
    
    if input_type == "raw":
        # Enhanced processing for real photos
        img_array = preprocess_real_image(img_array)
    else:
        # Standard processing for MNIST-style
        img_array = cv2.resize(img_array, (28, 28))
    
    # Final normalization
    img_array = img_array.astype('float32') / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    
    return img_array

def preprocess_real_image(img_array):
    """
    Advanced preprocessing for real camera images
    """
    # Step 1: Resize maintaining aspect ratio
    height, width = img_array.shape
    size = max(height, width)
    
    # Create square canvas
    square_img = np.zeros((size, size), dtype=np.uint8)
    
    # Center the digit
    y_start = (size - height) // 2
    x_start = (size - width) // 2
    square_img[y_start:y_start+height, x_start:x_start+width] = img_array
    
    # Step 2: Resize to 28x28
    img_resized = cv2.resize(square_img, (28, 28))
    
    # Step 3: Auto-contrast enhancement
    # Remove extreme dark/light pixels
    p2, p98 = np.percentile(img_resized, (2, 98))
    img_contrast = np.clip((img_resized - p2) * 255.0 / (p98 - p2), 0, 255).astype(np.uint8)
    
    # Step 4: Smart inversion detection
    # Count pixels in center vs edges to determine background
    center_region = img_contrast[10:18, 10:18]
    border_region = np.concatenate([
        img_contrast[0:5, :].flatten(),    # top border
        img_contrast[-5:, :].flatten(),    # bottom border  
        img_contrast[:, 0:5].flatten(),    # left border
        img_contrast[:, -5:].flatten()     # right border
    ])
    
    center_brightness = np.mean(center_region)
    border_brightness = np.mean(border_region)
    
    # If center is darker than borders, likely white digit on black background
    if center_brightness < border_brightness - 20:
        img_contrast = 255 - img_contrast
    
    # Step 5: Noise reduction
    img_clean = cv2.medianBlur(img_contrast, 3)
    
    # Step 6: Make digit thicker (helps with thin handwriting)
    kernel = np.ones((2,2), np.uint8)
    img_thick = cv2.dilate(img_clean, kernel, iterations=1)
    
    return img_thick

def create_enhanced_model():
    """
    Create a more robust model with regularization
    """
    model = models.Sequential([
        # First conv block with more filters
        layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),  # Increased dropout
        
        # Second conv block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(), 
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),  # Increased dropout
        
        # Third conv block for better feature extraction
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        # Classifier
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),  # High dropout to prevent overfitting
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    
    return model

def load_mnist_data():
    """Load MNIST dataset"""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    return (x_train, y_train), (x_test, y_test)

def train_enhanced_model():
    """
    Train with data augmentation for better real-world performance
    """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    # Load data
    print("Loading MNIST data...")
    (x_train, y_train), (x_test, y_test) = load_mnist_data()
    
    # Preprocess
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    
    print(f"Training data: {x_train.shape}")
    print(f"Test data: {x_test.shape}")
    
    # Data Augmentation - creates variations of training images
    datagen = ImageDataGenerator(
        rotation_range=10,      # Rotate images slightly
        width_shift_range=0.1,  # Shift horizontally
        height_shift_range=0.1, # Shift vertically
        zoom_range=0.1,         # Zoom in/out
        shear_range=0.1         # Skew images
    )
    
    model = create_enhanced_model()
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Starting enhanced training with data augmentation...")
    
    # Train with augmented data
    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=128),
        epochs=50,
        validation_data=(x_test, y_test),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
        ]
    )
    
    model.save('D:/ML_PROJECT/models/enhanced_digit_model.h5')
    print("✅ Enhanced model saved successfully!")
    return model

# Update your web app prediction function
def enhanced_predict(image_path):
    """Enhanced prediction for real images"""
    model = tf.keras.models.load_model('D:/ML_PROJECT/models/enhanced_digit_model.h5')
    
    image = Image.open(image_path)
    processed_array = enhanced_preprocess(image, input_type="raw")
    prediction = model.predict(processed_array)
    predicted_digit = np.argmax(prediction)
    confidence = np.max(prediction)
    
    return predicted_digit, confidence

if __name__ == "__main__":
    train_enhanced_model()