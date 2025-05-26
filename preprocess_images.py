import cv2
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

mask_images_folder = r"C:\Users\shubh\OneDrive\Desktop\CS 455\Face_Mask_Detection_Project\data"

def extract_face_features(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return None

    x, y, w, h = faces[0]
    face = gray[y:y+h, x:x+w]

    face_resized = cv2.resize(face, (64, 64))  # Resize to 64x64

    face_features = face_resized.flatten()
    return face_features

def create_dataset(mask_images_folder):
    features = []
    labels = []

    for subfolder in ['with_mask', 'without_mask']:
        subfolder_path = os.path.join(mask_images_folder, subfolder)

        for filename in os.listdir(subfolder_path):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                img_path = os.path.join(subfolder_path, filename)

                face_features = extract_face_features(img_path)
                if face_features is not None:
                    features.append(face_features)

                    if subfolder == 'with_mask':
                        labels.append(1)
                    else:
                        labels.append(0)

    return np.array(features), np.array(labels)

features, labels = create_dataset(mask_images_folder)

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

# Test the classifier
y_pred = svm.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save the classifier for future use
import joblib
joblib.dump(svm, 'mask_classifier.pkl')
