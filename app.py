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
OBJECT_FOLDER = "static/objects"
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

                faces_rects = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=8, minSize=(30, 30))

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


@app.route('/process_payment', methods=['POST'])
def process_payment():
    # Here you would integrate the actual payment processing code
    # For now, we simulate a successful payment
    user_id = request.form.get('user_id')  # You can pass this value with the form

    # Clear the cart after payment
    result = users_collection.update_one(
        {"_id": ObjectId(user_id)},
        {"$set": {"cart_items": []}}  # Clear the cart
    )

    # if result.modified_count == 1:
    #     # Redirect to a success page or show a success message
    #     # return redirect(url_for('payment_success'))
    # else:
    #     return "Error processing payment", 500
    return jsonify({"message": "Payment successful!"})

@app.route('/payment_success')
def payment_success():
    return "Payment successful! Your cart has been cleared."

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
    # faces_rects = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
 
    faces_rects = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=8, minSize=(30, 30))

    if len(faces_rects) == 0:
        os.remove(face_image_path)
        return jsonify({"message": "No face detected."}), 400

    user_data = None
    for (x, y, w, h) in faces_rects:
        face = gray[y:y + h, x:x + w]
        face = cv2.resize(face, (150, 150))  # Resize to match the training size

        try:
            label_id, confidence = recognizer.predict(face)
            folder_id = [label for label, idx in label_ids.items() if idx == label_id][0]
            user = users_collection.find_one({"_id": ObjectId(folder_id)})

            if user:
                user_data = {
                    "id": str(user["_id"]),
                    "person": user["name"],
                    "mobile": user["mobile"],
                    "cart_items": user.get("cart_items", []),
                    "confidence": round(confidence, 2)
                }
        except Exception as e:
            print(f"Error in prediction: {e}")
            os.remove(face_image_path)
            return jsonify({"message": "Error during recognition."}), 500

    if not user_data:
        os.remove(face_image_path)
        return jsonify({"message": "Face detected, but no match found in the database."}), 404

    # Process object images
    object_images = request.files.getlist('object_images')
    if not object_images:
        os.remove(face_image_path)
        return jsonify({"message": "No object images uploaded."}), 400

    object_image_paths = []
    for obj_img in object_images:
        obj_filename = secure_filename(obj_img.filename)
        obj_path = os.path.join(UPLOAD_FOLDER, obj_filename)
        obj_img.save(obj_path)
        object_image_paths.append(obj_path)

    object_images_files = []
    for img_path in object_image_paths:
        obj_file = open(img_path, 'rb')
        object_images_files.append(('image', (os.path.basename(img_path), obj_file, 'application/octet-stream')))

    try:
        response = requests.post('http://192.168.1.4:5001/predict', 
                                 files=object_images_files, 
                                 data={"user_data": user_data})
        if response.status_code == 200:
            object_detection_results = response.json()
        else:
            object_detection_results = {"error": "Object detection failed."}

    except Exception as e:
        print(f"Error during object detection API request: {e}")
        os.remove(face_image_path)
        return jsonify({"message": "Failed to contact object detection API."}), 500

    # Add detected objects to user's cart
    if isinstance(object_detection_results, list):
        for detected_object in object_detection_results:
            try:
                # Split detected_object into name and price
                object_name, object_price = detected_object.split('_')
                object_price = float(object_price)  # Convert price to float

                # Create a dictionary for the object
                object_data = {"name": object_name, "price": object_price}

                # Update user's cart in the database
                result = users_collection.update_one(
                    {"_id": ObjectId(user_data["id"])},
                    {"$push": {"cart_items": object_data}}
                )

                if result.matched_count == 0:
                    print("No user found with the given ID.")
                elif result.modified_count == 0:
                    print("Detected object already exists in cart or no update made.")

            except Exception as e:
                print(f"Error processing detected object '{detected_object}': {e}")
    else:
        print("Invalid object detection results received.")

    # Fetch updated user data from database
    updated_user = users_collection.find_one({"_id": ObjectId(user_data["id"])})
    user_data["cart_items"] = updated_user.get("cart_items", [])

    # Clean up temporary files
    os.remove(face_image_path)
    # for path in object_image_paths:
    #     os.remove(path)

    # Render the bill
    print("User Data for Bill:", user_data)
    return render_template("bill.html", user_data=user_data)




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
