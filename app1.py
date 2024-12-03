import os
import time
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from script import get_combined_features, model, scaler, get_landmarks, pairs, plot_landmarks, is_frontal_face, crop_face

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for uploading and analyzing the image
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Load the uploaded image
        img = cv2.imread(filepath)

        # Step 1: Detect face and landmarks
        landmarks = get_landmarks(img)
        if not landmarks:
            return render_template('index.html', error="No frontal face detected in the image.")

        # Step 2: Check if the face is frontal
        if not is_frontal_face(landmarks):
            return render_template('index.html', error="The face is not frontal. Please try another picture.")

        # Step 3: Crop the face with padding
        cropped_image = crop_face(img, landmarks, padding_percentage=0.05)

        # Reapply face detection on the cropped image
        cropped_landmarks = get_landmarks(cropped_image)
        if not cropped_landmarks:
            return render_template('index.html', error="Unable to detect landmarks on the cropped image.")

        # Step 4: Analyze the cropped image
        start_time = time.time()
        features = get_combined_features(cropped_image, pairs)
        if features:
            features = np.nan_to_num(features)
            features = scaler.transform([features])
            probabilities = model.predict_proba(features)[0]
            prediction = model.predict(features)[0]
            end_time = time.time()

            # Inference time
            inference_time = end_time - start_time

            # Prediction result
            label = "Down Syndrome" if prediction == 1 else "Healthy"

            # Save the cropped and landmark plot
            cropped_filename = f"cropped_{filename}"
            cropped_filepath = os.path.join(app.config['UPLOAD_FOLDER'], cropped_filename)
            cv2.imwrite(cropped_filepath, cropped_image)

            plot_path = os.path.join(app.config['UPLOAD_FOLDER'], f"landmarks_{filename}")
            plot_landmarks(cropped_image.copy(), cropped_landmarks, plot_path)

            return render_template(
                'index.html',
                label=label,
                confidence_healthy=f"{probabilities[0]:.2f}",
                confidence_ds=f"{probabilities[1]:.2f}",
                inference_time=f"{inference_time:.4f} seconds",
                image_path=url_for('static', filename=f'uploads/{cropped_filename}'),
                plot_path=url_for('static', filename=f'uploads/landmarks_{filename}')
            )
        else:
            return render_template('index.html', error="No features could be extracted from the image.")

    return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)
