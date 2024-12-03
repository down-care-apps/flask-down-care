import os
import time
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from script import (
    get_separated_features,
    model,
    scaler,
    get_landmarks,
    pairs,
    plot_landmarks,
    is_frontal_face,
    crop_face,
    extract_lbp_from_patches,
    extract_geometric_features,
    get_separated_features,
)

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

        # Step 4: Reapply detection on the cropped image
        cropped_landmarks = get_landmarks(cropped_image)
        if not cropped_landmarks:
            return render_template('index.html', error="Unable to detect landmarks on the cropped image.")

        # Step 5: Extract LBP and geometric features
        lbp_features, geom_features = get_separated_features(cropped_image, pairs)

        # Step 6: Concatenate features with labels for display
        combined_features_with_labels = []
        lbp_array = np.hstack(list(lbp_features.values()))
        geom_array = np.array(list(geom_features.values()))

        # Add LBP features with labels
        for landmark, lbp in zip(lbp_features.keys(), lbp_array):
            combined_features_with_labels.append(f"{landmark}: {lbp}")

        # Add geometric features with labels
        for pair, distance in zip(geom_features.keys(), geom_array):
            combined_features_with_labels.append(f"{pair}: {distance}")

        # Step 7: Transform features and make prediction
        start_time = time.time()
        combined_features = np.hstack([lbp_array, geom_array])
        combined_features = np.nan_to_num(combined_features)
        transformed_features = scaler.transform([combined_features])
        probabilities = model.predict_proba(transformed_features)[0]
        prediction = model.predict(transformed_features)[0]
        end_time = time.time()

        # Save the cropped and landmark plot
        cropped_filename = f"cropped_{filename}"
        cropped_filepath = os.path.join(app.config['UPLOAD_FOLDER'], cropped_filename)
        cv2.imwrite(cropped_filepath, cropped_image)

        plot_path = os.path.join(app.config['UPLOAD_FOLDER'], f"landmarks_{filename}")
        plot_landmarks(cropped_image.copy(), cropped_landmarks, plot_path)

        label = "Down Syndrome" if prediction == 1 else "Healthy"
        inference_time = f"{end_time - start_time:.4f} seconds"

        # Render template with all features and results
        return render_template(
            'index.html',
            label=label,
            confidence_healthy=f"{probabilities[0]:.2f}",
            confidence_ds=f"{probabilities[1]:.2f}",
            inference_time=inference_time,
            lbp_features=lbp_features,
            geom_features=geom_features,
            combined_features=combined_features_with_labels,
            image_path=url_for('static', filename=f'uploads/{cropped_filename}'),
            plot_path=url_for('static', filename=f'uploads/landmarks_{filename}')
        )

    return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)
