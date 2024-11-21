# import os
# import cv2
# import numpy as np
# from flask import Flask, request, jsonify, render_template
# from werkzeug.utils import secure_filename

# app = Flask(__name__)

# # Path to dataset directory
# DATASET_DIR = "dataset"
# UPLOAD_FOLDER = "static/uploads"
# os.makedirs(DATASET_DIR, exist_ok=True)
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# # Initialize Haar Cascade face detector
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# # Initialize LBPH face recognizer
# recognizer = cv2.face.LBPHFaceRecognizer_create()

# # Prepare training data function (same as before)
# def prepare_training_data(data_folder_path):
#     faces = []
#     labels = []
    
#     dirs = os.listdir(data_folder_path)

#     for dir_name in dirs:
#         if not dir_name.startswith("."):
#             label = dir_name
#             subject_dir_path = os.path.join(data_folder_path, dir_name)
#             subject_images_names = os.listdir(subject_dir_path)

#             for image_name in subject_images_names:
#                 if image_name.startswith("."):
#                     continue

#                 image_path = os.path.join(subject_dir_path, image_name)
#                 image = cv2.imread(image_path)
#                 gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#                 faces_rects = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

#                 for (x, y, w, h) in faces_rects:
#                     face = gray[y:y + h, x:x + w]
#                     faces.append(face)
#                     labels.append(label)

#     return faces, labels

# # Train recognizer
# def train_recognizer():
#     faces, labels = prepare_training_data(DATASET_DIR)
#     label_ids = {name: idx for idx, name in enumerate(set(labels))}
#     numeric_labels = np.array([label_ids[label] for label in labels])
#     recognizer.train(faces, numeric_labels)
#     return label_ids

# # Train the model
# label_ids = train_recognizer()

# # Endpoint to upload images to dataset (for training)
# @app.route('/upload', methods=['POST'])
# def upload_images():
#     person_name = request.form['name'].lower()
#     images = request.files.getlist('images')

#     person_dir = os.path.join(DATASET_DIR, person_name)
    
#     if not os.path.exists(person_dir):
#         os.makedirs(person_dir)

#     for img in images:
#         img_filename = secure_filename(img.filename)
#         img.save(os.path.join(person_dir, img_filename))

#     # Re-train the recognizer after adding new images
#     global label_ids
#     label_ids = train_recognizer()

#     return jsonify({"message": f"Images for {person_name} uploaded and model retrained!"})

# # Endpoint to recognize a face from an image
# @app.route('/recognize', methods=['POST'])
# def recognize():
#     img = request.files['image']
#     img_filename = secure_filename(img.filename)
#     img_path = os.path.join(UPLOAD_FOLDER, img_filename)
#     img.save(img_path)

#     frame = cv2.imread(img_path)
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     faces_rects = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

#     for (x, y, w, h) in faces_rects:
#         face = gray[y:y + h, x:x + w]
#         label_id, confidence = recognizer.predict(face)

#         label = [name for name, idx in label_ids.items() if idx == label_id][0]

#         return jsonify({"person": label, "confidence": round(confidence, 2)})

#     return jsonify({"message": "No face detected."})

# # Webpage to upload images for training
# @app.route('/up')
# def upload_page():
#     return render_template('upload.html')

# # Webpage to recognize a face
# @app.route('/re')
# def recognize_page():
#     return render_template('recognize.html')

# if __name__ == '__main__':
#     app.run(debug=True)

import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from pymongo import MongoClient
from bson.objectid import ObjectId
import requests

app = Flask(__name__)

# Path to dataset directory
DATASET_DIR = "./dataset"
UPLOAD_FOLDER = "static/uploads"
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize MongoDB client
client = MongoClient("mongodb+srv://yashshinde1990:X70in5zgdwB3DMrc@cluster0.tqwry.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")  # Adjust connection string as needed
db = client["inventory_management"]
users_collection = db["users"]

# Initialize Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Prepare training data function
def prepare_training_data(data_folder_path):
    faces = []
    labels = []

    dirs = os.listdir(data_folder_path)

    for dir_name in dirs:
        if not dir_name.startswith("."):
            label = dir_name  # Folder name is the primary key (_id)
            subject_dir_path = os.path.join(data_folder_path, dir_name)
            subject_images_names = os.listdir(subject_dir_path)

            for image_name in subject_images_names:
                if image_name.startswith("."):
                    continue

                image_path = os.path.join(subject_dir_path, image_name)
                image = cv2.imread(image_path)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                faces_rects = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

                for (x, y, w, h) in faces_rects:
                    face = gray[y:y + h, x:x + w]
                    faces.append(face)
                    labels.append(label)

    return faces, labels

def train_recognizer():
    faces, labels = prepare_training_data(DATASET_DIR)
    label_ids = {label: idx for idx, label in enumerate(set(labels))}
    numeric_labels = np.array([label_ids[label] for label in labels])
    recognizer.train(faces, numeric_labels)
    return label_ids

# Train the model
label_ids = train_recognizer()

# Endpoint to upload images to dataset and add user details to MongoDB
@app.route('/upload', methods=['POST'])
def upload_images():
    person_name = request.form['name'].lower()
    mobile_number = request.form['mobile']
    images = request.files.getlist('images')

    # Create a new user record in MongoDB and retrieve the primary key
    user = {
        "name": person_name,
        "mobile": mobile_number,
        "cart_items": []  # Initialize with an empty cart
    }
    result = users_collection.insert_one(user)
    user_id = str(result.inserted_id)  # MongoDB ObjectId as primary key

    # Create a folder named after the primary key
    person_dir = os.path.join(DATASET_DIR, user_id)
    if not os.path.exists(person_dir):
        os.makedirs(person_dir)

    # Save images to the user's folder
    for img in images:
        img_filename = secure_filename(img.filename)
        img.save(os.path.join(person_dir, img_filename))

    # Re-train the recognizer after adding new images
    global label_ids
    label_ids = train_recognizer()

    return jsonify({"message": f"User {person_name} added with ID {user_id}, and model retrained!"})



@app.route('/recognize', methods=['POST'])
def recognize():
    # Save the uploaded face image
    face_image = request.files.get('face_image')
    if not face_image:
        return jsonify({"message": "No face image uploaded."}), 400

    face_image_filename = secure_filename(face_image.filename)
    face_image_path = os.path.join(UPLOAD_FOLDER, face_image_filename)
    face_image.save(face_image_path)

    # Read and preprocess the face image
    frame = cv2.imread(face_image_path)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using Haar Cascade
    faces_rects = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces_rects) == 0:
        os.remove(face_image_path)
        return jsonify({"message": "No face detected."}), 400

    # Predict for each detected face
    user_data = None
    for (x, y, w, h) in faces_rects:
        face = gray[y:y + h, x:x + w]
        face = cv2.resize(face, (150, 150))  # Resize to match the training size
        
        try:
            # Predict the label ID and confidence
            label_id, confidence = recognizer.predict(face)

            # Get folder name (primary key) from label ID
            folder_id = [label for label, idx in label_ids.items() if idx == label_id][0]

            # Fetch user details from MongoDB using the folder name as primary key
            user = users_collection.find_one({"_id": ObjectId(folder_id)})
            # print(user)

            if user:
                user_data = {
                    "id": str(user["_id"]),
                    "person": user["name"],
                    "mobile": user["mobile"],
                    "cart_items": user.get("cart_items", []),
                    "confidence": round(confidence, 2)
                }
                  # Stop after the first match
            print(user_data)
        except Exception as e:
            print(f"Error in prediction: {e}")
            os.remove(face_image_path)
            return jsonify({"message": "Error during recognition."}), 500

    # If no user is found or face matching fails
    if not user_data:
        os.remove(face_image_path)
        return jsonify({"message": "Face detected, but no match found in the database."}), 404

    # Save the object images to process them
    object_images = request.files.getlist('object_images')
    if not object_images:
        os.remove(face_image_path)
        return jsonify({"message": "No object images uploaded."}), 400

    # Prepare object images for API call
    object_image_paths = []
    for obj_img in object_images:
        obj_filename = secure_filename(obj_img.filename)
        obj_path = os.path.join(UPLOAD_FOLDER, obj_filename)
        obj_img.save(obj_path)
        object_image_paths.append(obj_path)

    # Now, make a request to the external object detection API (127.0.0.1:5001/predict)
    object_images_files = []
    # for img_path in object_image_paths:
    #     with open(img_path, 'rb') as f:
    #         object_images_files.append(('image', (os.path.basename(img_path), f)))

    for img_path in object_image_paths:
        # Open the file and keep the file pointers open until the request is complete
        obj_file = open(img_path, 'rb')
        object_images_files.append(('image', (os.path.basename(img_path), obj_file, 'application/octet-stream')))

    try:
        # Make the POST request to the object detection API
        response = requests.post('http://127.0.0.1:5001/predict', 
                                 files=object_images_files, 
                                 data={"user_data": user_data})

        if response.status_code == 200:
            object_detection_results = response.json()
            print(object_detection_results)
        else:
            object_detection_results = {"error": "Object detection failed."}

    except Exception as e:
        print(f"Error during object detection API request: {e}")
        os.remove(face_image_path)
        for path in object_image_paths:
            os.remove(path)
        return jsonify({"message": "Failed to contact object detection API."}), 500

    # Clean up uploaded images
    os.remove(face_image_path)
    # for path in object_image_paths:
    #     os.remove(path)

    # Return the combined response
    return jsonify({
        "user_data": user_data,
        "object_detection_results": object_detection_results
    })


# Webpage to upload images for training
@app.route('/up')
def upload_page():
    return render_template('upload.html')

# Webpage to recognize a face
@app.route('/re')
def recognize_page():
    return render_template('recognize.html')

if __name__ == '__main__':
    app.run(debug=True)

# import requests
# import os
# import cv2
# from flask import request, jsonify
# from werkzeug.utils import secure_filename
# from pymongo import MongoClient
# from bson.objectid import ObjectId

# UPLOAD_FOLDER = "static/uploads"
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# # Assuming the face recognizer and label ids are already defined globally
# recognizer = cv2.face.LBPHFaceRecognizer_create()
# Endpoint to recognize a face from an image and fetch user details
# @app.route('/recognize', methods=['POST'])
# def recognize():
#     # Save the uploaded image
#     img = request.files['image']
#     img_filename = secure_filename(img.filename)
#     img_path = os.path.join(UPLOAD_FOLDER, img_filename)
#     img.save(img_path)

#     # Read and preprocess the image
#     frame = cv2.imread(img_path)
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Detect faces using Haar Cascade
#     faces_rects = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

#     if len(faces_rects) == 0:
#         return jsonify({"message": "No face detected."})

#     # Predict for each detected face
#     for (x, y, w, h) in faces_rects:
#         face = gray[y:y + h, x:x + w]
#         face = cv2.resize(face, (150, 150))  # Resize to match the training size
        
#         try:
#             # Predict the label ID and confidence
#             label_id, confidence = recognizer.predict(face)

#             # Get folder name (primary key) from label ID
#             folder_id = [label for label, idx in label_ids.items() if idx == label_id][0]

#             # Fetch user details from MongoDB using the folder name as primary key
#             user = users_collection.find_one({"_id": ObjectId(folder_id)})

#             if user:
#                 # Return user details if found
#                 userData = jsonify({
#                     "id": str(user["_id"]),
#                     "person": user["name"],
#                     "mobile": user["mobile"],
#                     "cart_items": user.get("cart_items", []),
#                     "confidence": round(confidence, 2)
#                 })
#                 return userData 

#         except Exception as e:
#             print(f"Error in prediction: {e}")
#             return jsonify({"message": "Error during recognition."})
#         finally:
#             os.remove(img_path)
            

#     return jsonify({"message": "Face detected, but no match found in the database. " + folder_id})