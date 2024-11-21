import cv2
import numpy as np
import joblib
import dlib
from skimage.feature import local_binary_pattern
from matplotlib import pyplot as plt

# Load the trained model and scaler
model_data = joblib.load('trained_model.pkl')
model = model_data['model']
scaler = model_data['scaler']

# Load Dlib facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Constants for LBP
RADIUS = 2
POINTS = 8 * RADIUS
METHOD = 'uniform'
PATCH_SIZE = 16

# List of specific landmarks to extract patches from
landmark_indices = [
    36, 39, 42, 45, 27, 30, 33, 31, 35, 51, 48, 54, 57, 68
]

# Define landmark pairs for geometric features
pairs = [
    (36, 39), (39, 42), (42, 45), (27, 30), (30, 33),
    (33, 31), (33, 35), (30, 31), (30, 35), (33, 51),
    (51, 48), (51, 54), (51, 57), (48, 57), (54, 57),
    (39, 68), (42, 68)
]

# Function to check if the face is frontal
def is_frontal_face(landmarks):
    left_eye = np.mean(np.array(landmarks[36:42]), axis=0)
    right_eye = np.mean(np.array(landmarks[42:48]), axis=0)
    nose_tip = np.array(landmarks[30])
    eye_distance = np.linalg.norm(left_eye - right_eye)
    nose_to_left_eye = np.linalg.norm(nose_tip - left_eye)
    nose_to_right_eye = np.linalg.norm(nose_tip - right_eye)
    symmetry_threshold = 0.15 * eye_distance
    return abs(nose_to_left_eye - nose_to_right_eye) < symmetry_threshold

# Function to extract landmarks
def get_landmarks(image_input):
    gray = cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)
        points = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(68)]
        midpoint_x = (points[21][0] + points[22][0]) // 2
        midpoint_y = (points[21][1] + points[22][1]) // 2
        points.append((midpoint_x, midpoint_y))
        if is_frontal_face(points):
            return points
    return None

# Function to extract patches
def extract_patches(image, landmarks, indices, patch_size=PATCH_SIZE):
    patches = []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for idx in indices:
        (x, y) = landmarks[idx]
        x_start = max(x - patch_size // 2, 0)
        y_start = max(y - patch_size // 2, 0)
        x_end = min(x + patch_size // 2, gray.shape[1])
        y_end = min(y + patch_size // 2, gray.shape[0])
        patch = gray[y_start:y_end, x_start:x_end]
        if patch.size > 0:
            patches.append(patch)
    return patches

# Function to extract LBP features
def extract_lbp_from_patches(patches, radius=RADIUS, points=POINTS, method=METHOD):
    lbp_features = []
    for patch in patches:
        lbp = local_binary_pattern(patch, points, radius, method)
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, points + 3), range=(0, points + 2))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-6)
        lbp_features.extend(hist)
    return lbp_features

# Function to extract geometric features
def extract_geometric_features(landmarks, pairs):
    geom_features = []
    for i, j in pairs:
        p1 = landmarks[i]
        p2 = landmarks[j]
        distance = np.linalg.norm(np.array(p1) - np.array(p2))
        geom_features.append(distance)
    return geom_features

# Function to extract combined features (LBP + Geometric)
def get_combined_features(image, pairs):
    landmarks = get_landmarks(image)
    if landmarks:
        patches = extract_patches(image, landmarks, landmark_indices)
        lbp_features = extract_lbp_from_patches(patches)
        geom_features = extract_geometric_features(landmarks, pairs)
        combined_features = lbp_features + geom_features
        return combined_features
    return None

# Function to plot landmarks and save the plot
def plot_landmarks(image, landmarks, save_path):
    for x, y in landmarks:
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
    for (i, j) in pairs:
        x1, y1 = landmarks[i]
        x2, y2 = landmarks[j]
        cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 1)
    cv2.imwrite(save_path, image)
