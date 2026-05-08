import os
import base64
import requests
import numpy as np
from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image
import io
import datetime
import json
import uuid
import traceback

from report import send_whatsapp_report

app = Flask(__name__, 
            static_folder='../frontend',
            template_folder='../frontend')
app.secret_key = 'seizureguard-secret-key-2024'
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Roboflow API Configuration
ROBOFLOW_API_KEY = "GzPWxtRmWmhIxgW3ULgh"  # Your API key
ROBOFLOW_MODEL_ID = "chb-mit-scalp-eeg-database-expo/1"
ROBOFLOW_API_URL = f"https://detect.roboflow.com/{ROBOFLOW_MODEL_ID}"

# Confidence threshold for seizure detection 
# Based on actual API response where seizure confidence is around 0.000057%
SEIZURE_THRESHOLD = 0.000001  # 0.0001% threshold - much lower to detect actual seizure signals

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image_for_api(image_path):
    """
    Preprocess image for Roboflow API
    """
    try:
        # Open and resize image
        img = Image.open(image_path)
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize to expected size (model specific)
        img = img.resize((640, 640))
        
        # Convert to base64
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return img_base64
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def call_roboflow_api(image_base64):
    """
    Call Roboflow API for prediction
    """
    try:
        # Prepare headers and data
        headers = {
            "Content-Type": "application/x-www-form-urlencoded"
        }
        
        # Make API request
        response = requests.post(
            f"{ROBOFLOW_API_URL}?api_key={ROBOFLOW_API_KEY}",
            data=image_base64,
            headers=headers
        )
        
        print(f"API Response Status: {response.status_code}")
        
        if response.status_code == 200:
            response_json = response.json()
            print(f"API Response received successfully")
            return response_json
        else:
            print(f"API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"Error calling Roboflow API: {e}")
        return None

def parse_roboflow_response(api_response):
    """
    Parse Roboflow API response to extract prediction and confidence
    Format from actual API:
    {
        "predictions": {
            "non-seizure": {"confidence": 0.9999, "class_id": 0},
            "seizure": {"confidence": 0.000057, "class_id": 1}
        },
        "predicted_classes": ["non-seizure"]
    }
    """
    try:
        # Default values
        is_seizure = False
        confidence = 0
        seizure_confidence = 0
        non_seizure_confidence = 0
        class_name = "unknown"
        all_predictions = {}
        
        if not api_response:
            return is_seizure, confidence, seizure_confidence, non_seizure_confidence, class_name, all_predictions
        
        # Get predictions dictionary
        if 'predictions' in api_response:
            all_predictions = api_response['predictions']
            
            # Extract individual confidences
            seizure_confidence = all_predictions.get('seizure', {}).get('confidence', 0)
            non_seizure_confidence = all_predictions.get('non-seizure', {}).get('confidence', 0)
            
            # Use threshold to determine seizure
            # If seizure confidence is greater than threshold, classify as seizure
            if seizure_confidence > SEIZURE_THRESHOLD:
                is_seizure = True
                confidence = seizure_confidence
                class_name = 'seizure'
                print(f"🔴 SEIZURE DETECTED! Confidence: {seizure_confidence*100:.6f}%")
            else:
                is_seizure = False
                confidence = non_seizure_confidence
                class_name = 'non-seizure'
                print(f"🟢 NO SEIZURE. Non-seizure confidence: {non_seizure_confidence*100:.6f}%")
            
            # Also check predicted_classes array as backup
            if 'predicted_classes' in api_response and len(api_response['predicted_classes']) > 0:
                api_class = api_response['predicted_classes'][0]
                print(f"API predicted class: {api_class}")
        
        return is_seizure, confidence * 100, seizure_confidence * 100, non_seizure_confidence * 100, class_name, all_predictions
        
    except Exception as e:
        print(f"Error parsing API response: {e}")
        return False, 0, 0, 0, "unknown", {}

def generate_insights(is_seizure, confidence, seizure_conf, non_seizure_conf, class_name):
    """
    Generate clinical insights based on predictions
    """
    insights = []
    
    if is_seizure:
        insights.append("⚠️ ABNORMAL PATTERN DETECTED")
        insights.append("🔴 Epileptiform activity detected in EEG signal")
        if confidence > 0.001:  # 0.001% threshold for high confidence seizure
            insights.append("🚨 CRITICAL: High probability of seizure activity")
            insights.append("⚡ Immediate medical attention required")
        elif confidence > 0.0001:  # 0.0001% threshold for moderate confidence
            insights.append("⚠️ Moderate probability - Urgent consultation recommended")
        else:
            insights.append("📋 Low confidence detection - Further evaluation needed")
    else:
        insights.append("✅ NORMAL PATTERN DETECTED")
        insights.append("🟢 No epileptiform activity identified")
        insights.append("📊 Brain wave patterns within normal range")
    
    # Add confidence details with scientific notation for very small numbers
    if seizure_conf < 0.001:
        insights.append(f"📈 Seizure confidence: {seizure_conf:.6f}% ({seizure_conf/100:.2e})")
    else:
        insights.append(f"📈 Seizure confidence: {seizure_conf:.4f}%")
    
    if non_seizure_conf < 0.001:
        insights.append(f"📉 Non-seizure confidence: {non_seizure_conf:.6f}% ({non_seizure_conf/100:.2e})")
    else:
        insights.append(f"📉 Non-seizure confidence: {non_seizure_conf:.4f}%")
    
    # Add classification
    insights.append(f"🏷️ Classification: {class_name.replace('-', ' ').title()}")
    
    # Add threshold info with scientific notation
    insights.append(f"⚙️ Detection threshold: {SEIZURE_THRESHOLD*100:.6f}% ({SEIZURE_THRESHOLD:.2e})")
    
    # Add recommendations based on confidence
    if is_seizure:
        if confidence > 0.001:
            insights.append("🏥 EMERGENCY: Contact emergency services immediately")
        elif confidence > 0.0001:
            insights.append("🏥 URGENT: Schedule neurologist appointment today")
        else:
            insights.append("🏥 Schedule follow-up with neurologist within a week")
    else:
        insights.append("🏥 Continue regular monitoring as recommended by your doctor")
    
    # Add model info
    insights.append("🧠 Model: CHB-MIT Scalp EEG Database (ResNet101)")
    insights.append("📊 Training data: 6,579 images")
    
    return insights

@app.route('/')
def index():
    """Serve the home page"""
    return render_template('index.html')

@app.route('/upload')
def upload_page():
    """Serve the upload page"""
    return render_template('upload.html')

@app.route('/result')
def result_page():
    """Serve the result page"""
    return render_template('result.html')

@app.route('/about')
def about_page():
    """Serve the about page"""
    return render_template('about.html')

@app.route('/features')
def features_page():
    """Serve the features page"""
    return render_template('features.html')

@app.route('/how-it-works')
def how_it_works_page():
    """Serve the how it works page"""
    return render_template('how-it-works.html')

@app.route('/contact')
def contact_page():
    """Serve the contact page"""
    return render_template('contact.html')

@app.route('/login')
def login_page():
    """Serve the login page"""
    return render_template('login.html')

@app.route('/signup')
def signup_page():
    """Serve the signup page"""
    return render_template('signup.html')

@app.route('/demo')
def demo_page():
    """Serve the demo page"""
    return render_template('demo.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    """
    Endpoint to analyze uploaded EEG image using Roboflow API
    """
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        # Check if file is empty
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check file type
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed. Please upload an image file (PNG, JPG, JPEG, GIF, BMP)'}), 400
        
        # Check file size
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        
        if file_size > MAX_FILE_SIZE:
            return jsonify({'error': f'File size exceeds {MAX_FILE_SIZE//(1024*1024)}MB limit'}), 400
        
        # Generate unique filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        filename = secure_filename(file.filename)
        extension = filename.rsplit('.', 1)[1].lower() if '.' in filename else 'jpg'
        unique_filename = f"{timestamp}_{unique_id}.{extension}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        # Save file
        file.save(filepath)
        print(f"File saved: {filepath}")
        
        # Preprocess image for API
        image_base64 = preprocess_image_for_api(filepath)
        
        if not image_base64:
            return jsonify({'error': 'Error preprocessing image'}), 500
        
        # Call Roboflow API
        api_response = call_roboflow_api(image_base64)
        
        if not api_response:
            return jsonify({'error': 'Error calling analysis API'}), 500
        
        # Parse API response with threshold
        is_seizure, confidence, seizure_conf, non_seizure_conf, class_name, all_predictions = parse_roboflow_response(api_response)
        
        # Generate insights
        insights = generate_insights(is_seizure, confidence, seizure_conf, non_seizure_conf, class_name)
        
        # Prepare result with appropriate formatting for very small numbers
        result = {
            'prediction': 'Seizure Detected' if is_seizure else 'No Seizure',
            'confidence': f'{confidence:.6f}%' if confidence < 0.001 else f'{confidence:.2f}%',
            'probability': confidence / 100,
            'class_name': class_name,
            'seizure_confidence': f'{seizure_conf:.6f}%' if seizure_conf < 0.001 else f'{seizure_conf:.4f}%',
            'non_seizure_confidence': f'{non_seizure_conf:.6f}%' if non_seizure_conf < 0.001 else f'{non_seizure_conf:.4f}%',
            'seizure_confidence_raw': seizure_conf / 100,
            'non_seizure_confidence_raw': non_seizure_conf / 100,
            'threshold': f'{SEIZURE_THRESHOLD*100:.6f}%',
            'threshold_raw': SEIZURE_THRESHOLD,
            'status': 'success',
            'filename': unique_filename,
            'insights': insights,
            'analysis_id': unique_id,
            'timestamp': datetime.datetime.now().isoformat(),
            'model_info': {
                'name': 'CHB-MIT Scalp EEG Database',
                'version': '1',
                'type': 'ResNet101 Multi-label Classification',
                'trained_on': '6,579 images',
                'classes': ['seizure', 'non-seizure'],
                'threshold': f'{SEIZURE_THRESHOLD*100:.6f}%'
            },
            'api_response': api_response
        }

        # Send WhatsApp report (commented out for now)
        # send_whatsapp_report(
        #     result['prediction'],
        #     result['confidence'],
        #     result['seizure_confidence'],
        #     result['non_seizure_confidence']
        # )
        
        print(f"\n{'='*50}")
        print(f"Analysis complete: {result['prediction']}")
        print(f"Seizure confidence: {seizure_conf:.6f}% ({seizure_conf/100:.2e})")
        print(f"Non-seizure confidence: {non_seizure_conf:.6f}% ({non_seizure_conf/100:.2e})")
        print(f"Threshold used: {SEIZURE_THRESHOLD*100:.6f}% ({SEIZURE_THRESHOLD:.2e})")
        print(f"{'='*50}\n")
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Error in analyze_image: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/threshold', methods=['POST'])
def update_threshold():
    """Update the seizure detection threshold"""
    try:
        data = request.get_json()
        new_threshold = data.get('threshold')
        
        if new_threshold and 0 <= new_threshold <= 1:
            global SEIZURE_THRESHOLD
            SEIZURE_THRESHOLD = new_threshold
            return jsonify({
                'status': 'success',
                'message': f'Threshold updated to {new_threshold*100:.6f}%',
                'threshold': new_threshold,
                'threshold_percent': f'{new_threshold*100:.6f}%'
            })
        else:
            return jsonify({'error': 'Invalid threshold value'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.datetime.now().isoformat(),
        'model': 'CHB-MIT Scalp EEG Database',
        'api_configured': bool(ROBOFLOW_API_KEY),
        'current_threshold': f'{SEIZURE_THRESHOLD*100:.6f}%',
        'threshold_raw': SEIZURE_THRESHOLD
    })

@app.route('/api/info', methods=['GET'])
def api_info():
    """API information endpoint"""
    return jsonify({
        'name': 'SeizureGuard AI API',
        'version': '1.0',
        'model': 'CHB-MIT Scalp EEG Database',
        'model_version': '1',
        'model_type': 'ResNet101 Multi-label Classification',
        'current_threshold': f'{SEIZURE_THRESHOLD*100:.6f}%',
        'threshold_raw': SEIZURE_THRESHOLD,
        'endpoints': {
            '/': 'GET - Main page',
            '/api/analyze': 'POST - Upload and analyze image',
            '/api/threshold': 'POST - Update detection threshold',
            '/api/health': 'GET - Health check',
            '/api/info': 'GET - API information'
        }
    })

if __name__ == '__main__':
    print("="*60)
    print("SeizureGuard AI - Flask Application Starting")
    print("="*60)
    print(f"Model: CHB-MIT Scalp EEG Database")
    print(f"API Key: {'Configured' if ROBOFLOW_API_KEY else 'Missing'}")
    print(f"API URL: {ROBOFLOW_API_URL}")
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print(f"Current threshold: {SEIZURE_THRESHOLD*100:.6f}% ({SEIZURE_THRESHOLD:.2e})")
    print(f"Server will run on: http://localhost:5000")
    print("="*60)
    app.run(debug=True, host='0.0.0.0', port=5000)