import cv2
import joblib

# Load the trained classifier
svm = joblib.load('mask_classifier.pkl')

# Open wqebcam
cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        # Extract the face from the frame
        face = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (64, 64))  # Resize face for consistency
        face_features = face_resized.flatten()

        # Predict if the face has a mask
        prediction = svm.predict([face_features])
        print(f"Prediction: {prediction}")  # Debugging: print the prediction

        if prediction != 1:
            label = "No Mask"
            # Draw a red rectangle around the face if No Mask detected
            color = (0, 0, 255)  # Red color
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        else:
            label = "Mask"
            # Draw a green circle around the face if Mask detected
            color = (0, 255, 0)  # Green color
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            print("Mask Detected")  # Print when mask is detected
            

        # Put the label near the face
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Show the frame with bounding boxes and labels
    cv2.imshow("Face Mask Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
