import os
import time
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import torch
from torchvision import models, transforms
from PIL import Image

app = Flask(__name__)
CORS(app)  # Enable CORS for Flutter app

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB max file size

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model (same logic as predict_room.py)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Resolve model path - try multiple possible locations
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
MODEL_PATH = os.path.join(_project_root, "room_classifier.pt")

# If model not found in project root, try same directory as script
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(_script_dir, "room_classifier.pt")

# Global variables for model (loaded once at startup)
model = None
class_names = None
transform = None

def load_model():
    """Load the trained model once at startup"""
    global model, class_names, transform
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    class_names = checkpoint["class_names"]
    
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(DEVICE)
    model.eval()
    
    # Image transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    print(f"✅ Model loaded successfully. Classes: {class_names}")

def predict_room(image_path):
    """Predict room type from image path"""
    print(f"[PREDICT_ROOM] Processing image: {image_path}")
    
    try:
        # Load and convert image
        print(f"[PREDICT_ROOM] Loading image...")
        image = Image.open(image_path).convert("RGB")
        print(f"[PREDICT_ROOM] Image size: {image.size}, Mode: {image.mode}")
        
        # Transform image
        print(f"[PREDICT_ROOM] Transforming image...")
        image_tensor = transform(image).unsqueeze(0).to(DEVICE)
        print(f"[PREDICT_ROOM] Tensor shape: {image_tensor.shape}, Device: {DEVICE}")
        
        # Run prediction
        print(f"[PREDICT_ROOM] Running model prediction...")
        with torch.no_grad():
            outputs = model(image_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)
        
        predicted_class = class_names[predicted.item()]
        confidence_value = round(confidence.item(), 3)
        
        print(f"[PREDICT_ROOM] Raw outputs: {outputs.cpu().numpy()}")
        print(f"[PREDICT_ROOM] Probabilities: {probs.cpu().numpy()}")
        print(f"[PREDICT_ROOM] Predicted class index: {predicted.item()}")
        print(f"[PREDICT_ROOM] Predicted class: {predicted_class}")
        print(f"[PREDICT_ROOM] Confidence: {confidence_value}")
        
        result = {
            "prediction": predicted_class,
            "confidence": confidence_value
        }
        
        return result
    except Exception as e:
        print(f"[PREDICT_ROOM] ERROR in prediction: {e}")
        print(f"[PREDICT_ROOM] Traceback:\n{traceback.format_exc()}")
        raise

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    print(f"[HEALTH] Health check request received from {request.remote_addr}")
    result = {"status": "healthy", "model_loaded": model is not None}
    print(f"[HEALTH] Response: {result}")
    return jsonify(result)

@app.route('/predict', methods=['POST'])
def predict():
    """Predict room type from uploaded image"""
    start_time = time.time()
    print(f"\n{'='*60}")
    print(f"[PREDICT] New prediction request received")
    print(f"[PREDICT] Client: {request.remote_addr}")
    print(f"[PREDICT] Headers: {dict(request.headers)}")
    
    filepath = None
    try:
        # Check if image file exists in request
        if 'image' not in request.files:
            print("[PREDICT] ERROR: No image file in request")
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        print(f"[PREDICT] File received: {file.filename}")
        print(f"[PREDICT] Content type: {file.content_type}")
        print(f"[PREDICT] Content length: {file.content_length if hasattr(file, 'content_length') else 'unknown'}")
        
        if file.filename == '':
            print("[PREDICT] ERROR: Empty filename")
            return jsonify({"error": "No file selected"}), 400
        
        if not allowed_file(file.filename):
            print(f"[PREDICT] ERROR: Invalid file type: {file.filename}")
            return jsonify({"error": "Invalid file type. Allowed: png, jpg, jpeg, gif, webp"}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        print(f"[PREDICT] Saving file to: {filepath}")
        file.save(filepath)
        
        # Check if file was saved
        if not os.path.exists(filepath):
            print(f"[PREDICT] ERROR: File was not saved to {filepath}")
            return jsonify({"error": "Failed to save uploaded file"}), 500
        
        file_size = os.path.getsize(filepath)
        print(f"[PREDICT] File saved successfully. Size: {file_size} bytes")
        
        # Predict room type
        print(f"[PREDICT] Starting prediction...")
        predict_start = time.time()
        result = predict_room(filepath)
        predict_time = time.time() - predict_start
        
        print(f"[PREDICT] Prediction completed in {predict_time:.3f}s")
        print(f"[PREDICT] Result: {result}")
        
        # Clean up uploaded file
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"[PREDICT] Temporary file cleaned up")
        
        total_time = time.time() - start_time
        print(f"[PREDICT] Total request time: {total_time:.3f}s")
        print(f"{'='*60}\n")
        
        return jsonify(result)
    
    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        print(f"[PREDICT] ERROR occurred: {error_msg}")
        print(f"[PREDICT] Traceback:\n{error_trace}")
        
        # Clean up file if it exists
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
                print(f"[PREDICT] Temporary file cleaned up after error")
            except:
                pass
        
        total_time = time.time() - start_time
        print(f"[PREDICT] Total request time (with error): {total_time:.3f}s")
        print(f"{'='*60}\n")
        
        return jsonify({"error": error_msg}), 500

if __name__ == '__main__':
    try:
        load_model()
        print("🚀 Starting Flask server...")
        # Run on all interfaces so Flutter can connect
        app.run(host='0.0.0.0', port=5000, debug=True)
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        raise

