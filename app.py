import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import cv2
import time
import numpy as np
from firebase_admin import credentials, initialize_app, storage
import datetime
from script import get_combined_features, model, scaler, get_landmarks, pairs, plot_landmarks, is_frontal_face, crop_face
import requests
from urllib.parse import urlparse

# Initialize Firebase with your credentials
cred = credentials.Certificate('./service-account.json')
firebase_app = initialize_app(cred, {
    'storageBucket': 'mobile-pbl-app.firebasestorage.app'
})

def download_image_from_url(url):
    """ 
    Download an image from a URL and save it locally
    
    Args:
        url (str): URL of the image to download
    
    Returns:
        str: Local path where the image was saved
        None: If download fails
    """
    try:
        # Extract filename from URL
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)
        if not filename:
            filename = 'downloaded_image.jpg'
        
        # Ensure filename has an extension
        if not any(filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg']):
            filename += '.jpg'
            
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
        
        # Download the image
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Save the image
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    
        return filepath
    except Exception as e:
        print(f"Error downloading image: {str(e)}")
        return None

def upload_to_firebase(local_file_path, filename):
    """
    Upload a file to Firebase Storage and return its public URL
    
    Args:
        local_file_path (str): Path to the local file
        filename (str): Name to use for the file in Firebase Storage
    
    Returns:
        str: Public URL of the uploaded file
        None: If upload fails
    """
    try:
        bucket = storage.bucket()
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        destination_blob_name = f"landmarks/{timestamp}_{filename}"
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(local_file_path)
        blob.make_public()
        return blob.public_url
    except Exception as e:
        print(f"Error uploading to Firebase: {str(e)}")
        return None

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/api/analyze', methods=['POST'])
def analyze():
    filepath = None
    
    try:
        # Handle image URL
        if 'image_url' in request.form:
            image_url = request.form['image_url']
            filepath = download_image_from_url(image_url)
            if not filepath:
                return jsonify({'error': 'Failed to download image from URL'}), 400
        
        # Handle file upload
        elif 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected for uploading'}), 400
                
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
        else:
            return jsonify({'error': 'No file or image URL provided'}), 400

        # Process the image
        img = cv2.imread(filepath)

        landmarks = get_landmarks(img)
        if not landmarks:
            return jsonify({'error': 'No frontal face detected in the image'}), 400

        if not is_frontal_face(landmarks):
            return jsonify({'error': 'The face is not frontal. Please try another picture.'}), 400
        
        cropped_image = crop_face(img, landmarks, padding_percentage=0.05)
        cropped_landmarks = get_landmarks(cropped_image)
        if not cropped_landmarks:
            return jsonify({'error': 'Unable to detect landmarks on the cropped image'}), 400

        start_time = time.time()
        features = get_combined_features(cropped_image, pairs)
        if features:
            features = np.nan_to_num(features)
            features = scaler.transform([features])
            probabilities = model.predict_proba(features)[0]
            prediction = model.predict(features)[0]
            end_time = time.time()

            inference_time = end_time - start_time

            label = "Down Syndrome" if prediction == 1 else "Healthy"

            # Extract landmarks
        
            plot_path = os.path.join(app.config['UPLOAD_FOLDER'], f"landmarks_{os.path.basename(filepath)}")
            plot_landmarks(cropped_image.copy(), cropped_landmarks, plot_path)
            
            # Upload landmarks plot to Firebase
            firebase_url = upload_to_firebase(plot_path, f"landmarks_{os.path.basename(filepath)}")
            
            # Clean up local files
            os.remove(filepath)
            os.remove(plot_path)
            
            return jsonify({
                'label': label,
                'confidence': {
                    'healthy': f"{probabilities[0]:.2f}",
                    'down_syndrome': f"{probabilities[1]:.2f}"
                },
                'inference_time': f"{inference_time:.4f} seconds",
                'message': 'Analysis successful',
                'landmarks_url': firebase_url
            }), 200
        else:
            return jsonify({'error': 'No features could be extracted from the image'}), 400
            
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500
    finally:
        # Ensure cleanup of any remaining files
        if filepath and os.path.exists(filepath):
            os.remove(filepath)

if __name__ == '__main__':
    app.run(debug=True)