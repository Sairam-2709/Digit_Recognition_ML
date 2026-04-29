# web_app.py
from flask import Flask, render_template, request, jsonify
import numpy as np
from PIL import Image
import io
import base64
import os
import cv2
from tensorflow import keras

app = Flask(__name__)

# Create templates folder if it doesn't exist
current_dir = os.path.dirname(os.path.abspath(__file__))
templates_dir = os.path.join(current_dir, 'templates')
os.makedirs(templates_dir, exist_ok=True)

print(f"Templates directory: {templates_dir}")

# Check if template exists, if not create it
template_path = os.path.join(templates_dir, 'index.html')
if not os.path.exists(template_path):
    print("Creating index.html template...")
    # We'll create the HTML template automatically
    html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Digit Recognition Web App</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: Arial; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; padding: 20px; }
        .container { max-width: 800px; margin: 0 auto; background: white; border-radius: 15px; padding: 30px; box-shadow: 0 20px 40px rgba(0,0,0,0.1); }
        .header { text-align: center; margin-bottom: 30px; }
        .header h1 { color: #333; margin-bottom: 10px; }
        .upload-area { border: 3px dashed #667eea; border-radius: 10px; padding: 40px; text-align: center; margin: 20px 0; background: #f8f9fa; cursor: pointer; }
        .upload-area:hover { background: #e3f2fd; }
        .upload-btn { background: #667eea; color: white; border: none; padding: 12px 30px; border-radius: 25px; font-size: 16px; cursor: pointer; margin: 10px; }
        .options { margin: 20px 0; text-align: center; }
        .result { display: none; margin-top: 20px; padding: 20px; background: #f8f9fa; border-radius: 10px; }
        .images { display: flex; justify-content: center; gap: 20px; margin: 20px 0; }
        .image-box { text-align: center; }
        .image-box img { max-width: 200px; border: 2px solid #ddd; border-radius: 8px; }
        .prediction { text-align: center; font-size: 48px; font-weight: bold; color: #667eea; margin: 20px 0; }
        .loading { display: none; text-align: center; margin: 20px 0; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔢 Digit Recognition</h1>
            <p>Upload any handwritten digit image and get instant prediction!</p>
        </div>
        
        <div class="upload-area" id="uploadArea" onclick="document.getElementById('fileInput').click()">
            <div style="font-size: 48px; margin-bottom: 15px;">📁</div>
            <h3>Drag & Drop or Click to Upload</h3>
            <p>Supported formats: JPG, PNG, JPEG</p>
        </div>
        
        <input type="file" id="fileInput" style="display: none;" accept="image/*">
        
        <div class="options">
            <label><input type="radio" name="inputType" value="raw" checked> 📱 Raw Image (Mobile Photo)</label>
            <label style="margin-left: 20px;"><input type="radio" name="inputType" value="mnist"> 🎯 MNIST-style</label>
        </div>
        
        <div style="text-align: center;">
            <button class="upload-btn" onclick="uploadImage()">🚀 Predict Digit</button>
        </div>
        
        <div class="loading" id="loading">
            <div style="font-size: 24px; margin-bottom: 10px;">⏳</div>
            <p>Processing your image...</p>
        </div>
        
        <div class="result" id="result">
            <h2>Prediction Result</h2>
            <div class="images">
                <div class="image-box">
                    <img id="originalImg" src="" alt="Original">
                    <p>Original Image</p>
                </div>
                <div class="image-box">
                    <img id="processedImg" src="" alt="Processed">
                    <p>Processed Image</p>
                </div>
            </div>
            <div class="prediction" id="prediction">?</div>
            <p id="confidence" style="text-align: center; font-size: 18px;"></p>
        </div>
    </div>

    <script>
        let currentFile = null;
        
        // Drag and drop
        const uploadArea = document.getElementById('uploadArea');
        uploadArea.addEventListener('dragover', (e) => { e.preventDefault(); uploadArea.style.background = '#e3f2fd'; });
        uploadArea.addEventListener('dragleave', () => { uploadArea.style.background = '#f8f9fa'; });
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.style.background = '#f8f9fa';
            if (e.dataTransfer.files.length > 0) {
                currentFile = e.dataTransfer.files[0];
                alert('File selected: ' + currentFile.name);
            }
        });
        
        // File input
        document.getElementById('fileInput').addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                currentFile = e.target.files[0];
                alert('File selected: ' + currentFile.name);
            }
        });
        
        function uploadImage() {
            if (!currentFile) {
                alert('Please select an image file first!');
                return;
            }
            
            const formData = new FormData();
            formData.append('image', currentFile);
            formData.append('input_type', document.querySelector('input[name="inputType"]:checked').value);
            
            document.getElementById('loading').style.display = 'block';
            document.getElementById('result').style.display = 'none';
            
            fetch('/predict', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('loading').style.display = 'none';
                
                if (data.success) {
                    document.getElementById('originalImg').src = data.original_image;
                    document.getElementById('processedImg').src = data.processed_image;
                    document.getElementById('prediction').textContent = data.prediction;
                    document.getElementById('confidence').textContent = 'Confidence: ' + data.confidence + '%';
                    document.getElementById('result').style.display = 'block';
                } else {
                    alert('Error: ' + data.error);
                }
            })
            .catch(error => {
                document.getElementById('loading').style.display = 'none';
                alert('Error: ' + error);
            });
        }
    </script>
</body>
</html>'''
    
    with open(template_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print("✓ index.html template created automatically!")

# Load your trained model
try:
    model = keras.models.load_model('D:/ML_PROJECT/models/enhanced_digit_model.h5')
    print("✓ Model loaded successfully!")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    model = None

def preprocess_image(image, input_type="raw"):
    """
    Advanced preprocessing for raw camera images to match MNIST style.
    """
    import cv2
    import numpy as np

    # Convert to grayscale
    image = image.convert('L')
    img = np.array(image)

    if input_type == "raw":
        # 1️⃣ Remove shadows and normalize brightness
        img = cv2.GaussianBlur(img, (5, 5), 0)
        dilated = cv2.dilate(img, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated, 21)
        diff_img = 255 - cv2.absdiff(img, bg_img)
        norm_img = cv2.normalize(diff_img, None, 0, 255, cv2.NORM_MINMAX)

        # 2️⃣ Adaptive thresholding for binary conversion
        img_bin = cv2.adaptiveThreshold(
            norm_img, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 21, 5
        )

        # 3️⃣ Remove noise (tiny white dots)
        kernel = np.ones((3, 3), np.uint8)
        img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernel)

        # 4️⃣ Find largest contour (main digit)
        contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            if w > 10 and h > 10:
                digit = img_bin[y:y + h, x:x + w]
            else:
                digit = img_bin
        else:
            digit = img_bin

        # 5️⃣ Resize digit proportionally (20x20 max)
        h, w = digit.shape
        if h > w:
            new_h = 20
            new_w = int(w * (20 / h))
        else:
            new_w = 20
            new_h = int(h * (20 / w))
        digit = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # 6️⃣ Center digit on a 28x28 black canvas (MNIST format)
        canvas = np.zeros((28, 28), dtype=np.uint8)
        x_offset = (28 - new_w) // 2
        y_offset = (28 - new_h) // 2
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = digit

        # 7️⃣ Remove residual grays + normalize
        _, canvas = cv2.threshold(canvas, 50, 255, cv2.THRESH_BINARY)
        img_array = canvas.astype('float32') / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

    else:
        # MNIST-style input (already clean)
        image = image.resize((28, 28))
        img_array = np.array(image).astype('float32') / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

    return img_array

def convert_to_base64(image):
    """Convert PIL image to base64"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'success': False, 'error': 'Model not loaded'})
    
    try:
        file = request.files['image']
        input_type = request.form.get('input_type', 'raw')
        
        image = Image.open(io.BytesIO(file.read()))
        original_image = image.copy()
        
        processed_image = preprocess_image(image, input_type)
        
        prediction = model.predict(processed_image)
        predicted_digit = int(np.argmax(prediction))
        confidence = float(np.max(prediction))
        
        original_b64 = convert_to_base64(original_image)
        processed_display = Image.fromarray((processed_image[0, :, :, 0] * 255).astype('uint8'))
        processed_b64 = convert_to_base64(processed_display)
        
        return jsonify({
            'success': True,
            'prediction': predicted_digit,
            'confidence': round(confidence * 100, 2),
            'original_image': original_b64,
            'processed_image': processed_b64
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    print("🚀 Starting Digit Recognition Web App...")
    print("📧 Open your browser and go to: http://localhost:5000")
    app.run(debug=True, port=5000)
