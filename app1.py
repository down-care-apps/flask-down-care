import os
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from script import get_combined_features, model, scaler, get_landmarks, pairs, plot_landmarks

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

        # Process the image
        img = cv2.imread(filepath)
        img = cv2.resize(img, (300, 300))
        features = get_combined_features(img, pairs)

        if features:
            features = np.nan_to_num(features)
            features = scaler.transform([features])
            probabilities = model.predict_proba(features)[0]
            prediction = model.predict(features)[0]
            label = "Down Syndrome" if prediction == 1 else "Healthy"

            # Extract landmarks and plot them
            landmarks = get_landmarks(img)
            if landmarks:
                plot_path = os.path.join(app.config['UPLOAD_FOLDER'], f"landmarks_{filename}")
                plot_landmarks(img, landmarks, plot_path)  # Save the plot
                return render_template(
                    'index.html',
                    label=label,
                    confidence_healthy=f"{probabilities[0]:.2f}",
                    confidence_ds=f"{probabilities[1]:.2f}",
                    image_path=url_for('static', filename=f'uploads/{filename}'),
                    plot_path=url_for('static', filename=f'uploads/landmarks_{filename}')
                )
            else:
                return render_template('index.html', error="No frontal face detected in the image.")
        else:
            return render_template('index.html', error="No features could be extracted from the image.")

    return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)
