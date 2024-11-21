import cv2
import numpy as np
import os

# Initialize Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Directory containing training images (labeled)
dataset_dir = "dataset/"

def preprocess_face(face):
    face = cv2.resize(face, (150, 150))  # Ensure consistent size
    face = cv2.equalizeHist(face)        # Apply histogram equalization
    return face / 255.0                  # Normalize to [0, 1]


# Function to prepare training data
def prepare_training_data(data_folder_path):
    faces = []
    labels = []
    
    # Get the subdirectories in the dataset folder
    dirs = os.listdir(data_folder_path)

    for dir_name in dirs:
        if not dir_name.startswith("."):
            label = dir_name  # the person's name

            subject_dir_path = data_folder_path + "/" + dir_name
            subject_images_names = os.listdir(subject_dir_path)

            for image_name in subject_images_names:
                if image_name.startswith("."):
                    continue

                image_path = subject_dir_path + "/" + image_name
                image = cv2.imread(image_path)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # Detect face in the image
                faces_rects = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

                for (x, y, w, h) in faces_rects:
                    face = gray[y:y + h, x:x + w]
                    faces.append(face)
                    labels.append(label)

    return faces, labels

# Prepare training data
faces, labels = prepare_training_data(dataset_dir)

# Convert labels to numeric values because LBPH face recognizer needs numeric labels
label_ids = {name: idx for idx, name in enumerate(set(labels))}
numeric_labels = np.array([label_ids[label] for label in labels])

# Train the recognizer
recognizer.train(faces, numeric_labels)

# Function to predict and recognize face
def predict(frame, gray_frame):
    faces_rects = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in faces_rects:
        face = gray_frame[y:y + h, x:x + w]
        label_id, confidence = recognizer.predict(face)

        # Convert numeric label back to name
        label = [name for name, idx in label_ids.items() if idx == label_id][0]
        print(f"Detected: {label}, Confidence: {confidence}")

        # Draw rectangle around face and label it
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} ({round(confidence, 2)})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return frame

# Start webcam
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    output_frame = predict(frame, gray)

    cv2.imshow("Face Recognition", output_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
