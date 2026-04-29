✍️ Handwritten Digit Recognition using CNN
📌 Overview
This project builds a deep learning model to recognize handwritten digits (0–9) using a Convolutional Neural Network (CNN). The model is trained on the MNIST dataset and can also predict digits from custom images through a web interface.

🧠 Model Architecture
Convolutional Neural Network (CNN)
Multiple Conv2D + MaxPooling layers
Fully Connected (Dense) layers
Softmax output for multi-class classification

🚀 Features
Web interface for easy digit recognition
Support for both MNIST-style and real-world images
Real-time predictions with confidence scores
Drag & drop image upload
High accuracy on test dataset (~94%)

🛠️ Tech Stack
Python
TensorFlow / Keras
NumPy
OpenCV
Flask (for web app)

📂 Project Structure
handwritten-digit-recognition/
│
├── model/
│   └── cnn_model.h5
│
├── app/
│   ├── app.py
│   ├── templates/
│   │   └── index.html
│
├── static/
│   └── uploads/
│
├── notebooks/
│   └── training.ipynb
│
├── utils/
│   └── preprocessing.py
│
├── requirements.txt
└── README.md

📊 Dataset
MNIST Dataset (handwritten digits 0–9)
60,000 training images
10,000 testing images

⚙️ Model Training
Image normalization (pixel scaling)
Reshaping input for CNN
Model trained using categorical crossentropy loss
Optimizer: Adam

▶️ How to Run
1. Clone the repository
git clone https://github.com/your-username/handwritten-digit-recognition.git
cd handwritten-digit-recognition
2. Install dependencies
pip install -r requirements.txt
3. Run the application
python app/app.py
4. Open in browser
http://127.0.0.1:5000/

📷 Demo
(Add screenshots of your web app here)

🎯 Results
Achieved ~94% accuracy on test data
Successfully predicts handwritten digits from custom images

💡 Future Improvements
Improve accuracy with deeper CNN / data augmentation
Deploy model using cloud (GCP / AWS)
Add real-time drawing canvas for digit input

👨‍💻 Author
Sai ram Dasineni


