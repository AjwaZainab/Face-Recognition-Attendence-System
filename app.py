import os
import cv2
import numpy as np
import sqlite3
from datetime import datetime
import base64
import json
from flask import Flask, render_template, request, jsonify

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from mtcnn import MTCNN
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# --- Flask App Initialization ---
app = Flask(__name__)

# --- System Configuration ---
RAW_IMAGE_DIR = 'dataset/'
PROCESSED_IMAGE_DIR = 'processed_dataset/'
DB_FILE = 'attendance.db'
MODEL_PATH_H5 = 'facial_recognition_model.h5'
LABEL_ENCODER_PATH = 'label_encoder.npy'
USER_REGISTRY_PATH = 'user_registry.json'
IMG_SIZE = (160, 160)
EPOCHS = 50
BATCH_SIZE = 32

# --- Global Variables ---
detector = MTCNN()
model = None
labels = None # Will store registration numbers
user_registry = {} # Will store reg_no -> name mapping

# --- Helper Functions (Using SQLite and JSON) ---
def load_user_registry():
    """Loads the reg_no to name mapping from the JSON file."""
    global user_registry
    if os.path.exists(USER_REGISTRY_PATH):
        try:
            with open(USER_REGISTRY_PATH, 'r') as f:
                user_registry = json.load(f)
        except json.JSONDecodeError:
            print(f"WARN: Could not decode {USER_REGISTRY_PATH}. Starting with empty registry.")
            user_registry = {}
    else:
        user_registry = {}

def save_user_registry():
    """Saves the current user registry mapping to the JSON file."""
    try:
        with open(USER_REGISTRY_PATH, 'w') as f:
            json.dump(user_registry, f, indent=4)
    except IOError as e:
        print(f"ERROR: Could not save user registry to {USER_REGISTRY_PATH}: {e}")

def init_db():
    """Initializes the SQLite database and the attendance table."""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                reg_no TEXT NOT NULL,
                name TEXT NOT NULL,
                date TEXT NOT NULL,
                time TEXT NOT NULL,
                UNIQUE(reg_no, date)
            )''')
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        print(f"ERROR: Could not initialize database {DB_FILE}: {e}")


def get_model_status():
    """Checks if the model file exists."""
    return "Trained" if os.path.exists(MODEL_PATH_H5) else "Not Trained"

def get_recent_attendance():
    """Fetches the last 10 attendance records from SQLite."""
    records = []
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT reg_no, name, date, time FROM attendance ORDER BY id DESC LIMIT 10")
        records = cursor.fetchall()
        conn.close()
    except sqlite3.Error as e:
        print(f"ERROR: Could not get recent attendance from {DB_FILE}: {e}")
    return records

def load_recognition_model():
    """Loads the trained Keras model and the label encoder (reg_nos)."""
    global model, labels
    if get_model_status() == "Trained":
        try:
            model = load_model(MODEL_PATH_H5)
            labels = np.load(LABEL_ENCODER_PATH, allow_pickle=True)
            print("INFO: Recognition model and labels loaded.")
        except Exception as e:
            print(f"ERROR: Could not load model/label files: {e}")
            model = labels = None # Reset if loading fails

# --- Main Routes ---
@app.route('/')
def dashboard():
    return render_template('dashboard.html', model_status=get_model_status(), records=get_recent_attendance())

@app.route('/enroll')
def enroll_page():
    return render_template('enroll.html')

@app.route('/live_attendance')
def live_attendance_page():
    return render_template('live_attendance.html')

@app.route('/view_attendance')
def view_attendance():
    """Fetches and filters attendance records from SQLite."""
    all_records = []
    search_term = request.args.get('search', '').strip()
    search_date = request.args.get('date', '').strip()
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        query = "SELECT reg_no, name, date, time FROM attendance WHERE 1=1"
        params = []
        if search_term:
            query += " AND (name LIKE ? OR reg_no LIKE ?)"
            params.extend([f"%{search_term}%", f"%{search_term}%"])
        if search_date:
            query += " AND date = ?"
            params.append(search_date)
        query += " ORDER BY id DESC"
        cursor.execute(query, params)
        all_records = cursor.fetchall()
        conn.close()
    except sqlite3.Error as e:
        print(f"ERROR: Could not query attendance: {e}")

    return render_template('view_attendance.html', records=all_records, search_term=search_term, search_date=search_date)

@app.route('/manage_users')
def manage_users():
    """Displays users from the JSON registry and dataset folder."""
    registered_users = user_registry.copy()
    # Check dataset folder for users potentially missing from registry
    if os.path.exists(RAW_IMAGE_DIR):
        try:
            for reg_no in os.listdir(RAW_IMAGE_DIR):
                if os.path.isdir(os.path.join(RAW_IMAGE_DIR, reg_no)) and reg_no not in registered_users:
                    registered_users[reg_no] = "Unknown Name - Please Edit"
        except OSError as e:
            print(f"ERROR: Could not read dataset directory {RAW_IMAGE_DIR}: {e}")

    return render_template('manage_users.html', users=registered_users)

# --- API Endpoints ---
@app.route('/api/status')
def api_status():
    """Checks if the recognition model is loaded."""
    return jsonify({'model_loaded': model is not None and labels is not None})

@app.route('/api/capture', methods=['POST'])
def capture_image():
    """Saves captured image and updates the user registry JSON."""
    data = request.json
    person_name = data['name'].strip()
    reg_no = data['reg_no'].strip().upper()
    if not person_name or not reg_no:
        return jsonify({'status': 'error', 'message': 'Name and Registration No are required.'})

    # Update global registry and save it
    user_registry[reg_no] = person_name
    save_user_registry()

    # Save the image file
    person_folder = os.path.join(RAW_IMAGE_DIR, reg_no)
    os.makedirs(person_folder, exist_ok=True)
    try:
        header, encoded = data['image'].split(",", 1)
        img = cv2.imdecode(np.frombuffer(base64.b64decode(encoded), dtype=np.uint8), cv2.IMREAD_COLOR)
        count = len(os.listdir(person_folder))
        cv2.imwrite(os.path.join(person_folder, f"{reg_no}_{count + 1}.jpg"), img)
        return jsonify({'status': 'success', 'message': f'Image {count + 1} saved for {person_name} ({reg_no}).'})
    except Exception as e:
        print(f"ERROR: Could not save captured image: {e}")
        return jsonify({'status': 'error', 'message': f'Failed to save image: {e}'})


@app.route('/api/update_registry', methods=['POST'])
def update_registry():
    """Updates a user's name in the JSON registry."""
    data = request.json
    reg_no = data.get('reg_no')
    new_name = data.get('name')
    if not reg_no or not new_name:
        return jsonify({'success': False, 'message': 'Invalid data.'})

    user_registry[reg_no] = new_name
    save_user_registry()
    return jsonify({'success': True, 'message': f'Updated {reg_no} to {new_name}.'})

@app.route('/api/train', methods=['POST'])
def train_model_endpoint():
    """Trains the simple CNN model using data from disk."""
    # Step 1: Prepare the dataset
    print("INFO: Starting dataset preparation...")
    os.makedirs(PROCESSED_IMAGE_DIR, exist_ok=True)
    successfully_processed_users = set()
    for reg_no_folder in os.listdir(RAW_IMAGE_DIR):
        person_folder_in = os.path.join(RAW_IMAGE_DIR, reg_no_folder)
        person_folder_out = os.path.join(PROCESSED_IMAGE_DIR, reg_no_folder)
        if not os.path.isdir(person_folder_in): continue
        os.makedirs(person_folder_out, exist_ok=True)
        images_processed_for_this_user = 0
        for image_name in os.listdir(person_folder_in):
            try:
                img = cv2.imread(os.path.join(person_folder_in, image_name))
                if img is None: continue
                results = detector.detect_faces(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                if results:
                    x, y, w, h = results[0]['box']
                    face = img[y:y+h, x:x+w]
                    if face.size == 0: continue
                    resized_face = cv2.resize(face, IMG_SIZE)
                    cv2.imwrite(os.path.join(person_folder_out, image_name), resized_face)
                    images_processed_for_this_user += 1
            except Exception as e: print(f"WARN: Error processing {image_name}: {e}")
        if images_processed_for_this_user > 0:
            successfully_processed_users.add(reg_no_folder)

    if len(successfully_processed_users) < 2:
        return jsonify({'status': 'error', 'message': f'Training failed. Face detection was successful for only {len(successfully_processed_users)} user(s). Re-enroll with clearer pictures.'})
    print(f"INFO: Dataset preparation finished. Found faces for {len(successfully_processed_users)} users.")

    # Step 2: Load processed images and labels
    images, labels_list = [], []
    for reg_no_folder in successfully_processed_users:
        person_folder = os.path.join(PROCESSED_IMAGE_DIR, reg_no_folder)
        for image_name in os.listdir(person_folder):
            img = cv2.imread(os.path.join(person_folder, image_name))
            if img is not None:
                images.append(img.astype('float32') / 255.0) # Normalization
                labels_list.append(reg_no_folder) # Label is the reg_no

    # Step 3: Encode labels and split data
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels_list)
    np.save(LABEL_ENCODER_PATH, label_encoder.classes_) # Save reg_nos
    num_classes = len(label_encoder.classes_)
    categorical_labels = tf.keras.utils.to_categorical(encoded_labels, num_classes)
    X_train, X_test, y_train, y_test = train_test_split(np.array(images), categorical_labels, test_size=0.2, random_state=42, stratify=categorical_labels)

    # Step 4: Build the CNN Model (Simple & Effective)
    print("INFO: Building Simple CNN Model...")
    model_to_train = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        MaxPooling2D((2, 2)), Dropout(0.25),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)), Dropout(0.25),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)), Dropout(0.25),
        Flatten(),
        Dense(512, activation='relu'), Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model_to_train.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Step 5: Train the Model
    datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
    datagen.fit(X_train)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    print("INFO: Starting model training...")
    try:
        model_to_train.fit(
            datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
            steps_per_epoch=max(1, len(X_train) // BATCH_SIZE),
            validation_data=(X_test, y_test),
            epochs=EPOCHS,
            callbacks=[early_stopping],
            verbose=0
        )
    except Exception as e:
        print(f"ERROR: Exception during model training: {e}")
        return jsonify({'status': 'error', 'message': f'Model training failed: {e}'})


    # Step 6: Evaluate and Save
    try:
        loss, accuracy = model_to_train.evaluate(X_test, y_test, verbose=0)
        model_to_train.save(MODEL_PATH_H5)
        load_recognition_model() # Reload the newly trained model
    except Exception as e:
        print(f"ERROR: Exception during model evaluation or saving: {e}")
        return jsonify({'status': 'error', 'message': f'Model evaluation/saving failed: {e}'})

    print(f"INFO: Training finished. Final validation accuracy: {accuracy*100:.2f}%")
    return jsonify({'status': 'success', 'message': f'Model trained! Validation Accuracy: {accuracy*100:.2f}%'})

@app.route('/api/recognize', methods=['POST'])
def recognize_face():
    """Recognizes faces and logs attendance to SQLite."""
    if model is None or labels is None:
        return jsonify({'results': [], 'error': 'Model not loaded on the server.'})

    data = request.json
    try:
        header, encoded = data['image'].split(",", 1)
        frame = cv2.imdecode(np.frombuffer(base64.b64decode(encoded), dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"ERROR: Could not decode image data: {e}")
        return jsonify({'results': [], 'error': 'Invalid image data received.'})

    detected_faces = []
    try:
        results = detector.detect_faces(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        for result in results:
            x, y, w, h = result['box']
            face = frame[y:y+h, x:x+w]
            if face.size == 0: continue

            face_resized = cv2.resize(face, IMG_SIZE)
            face_array = face_resized.astype('float32') / 255.0 # Normalization
            face_array = np.expand_dims(face_array, axis=0)

            predictions = model.predict(face_array, verbose=0)
            confidence = np.max(predictions[0])

            person_reg_no = "Unknown"
            person_name = "Unknown"

            if confidence > 0.70:
                class_index = np.argmax(predictions[0])
                person_reg_no = labels[class_index] # Get reg_no from label encoder
                person_name = user_registry.get(person_reg_no, "Name Not Found") # Get name from JSON

                # --- Attendance Logic ---
                try:
                    conn = sqlite3.connect(DB_FILE)
                    cursor = conn.cursor()
                    today = datetime.now().strftime('%Y-%m-%d')
                    cursor.execute("SELECT id FROM attendance WHERE reg_no = ? AND date = ?", (person_reg_no, today))
                    if cursor.fetchone() is None:
                        current_time = datetime.now().strftime('%H:%M:%S')
                        cursor.execute("INSERT INTO attendance (reg_no, name, date, time) VALUES (?, ?, ?, ?)",
                                       (person_reg_no, person_name, today, current_time))
                        conn.commit()
                        print(f"SUCCESS: Marked attendance for {person_name} ({person_reg_no})")
                    conn.close()
                except sqlite3.Error as e:
                    print(f"ERROR: Database operation failed: {e}")
                # --- End Attendance Logic ---

            detected_faces.append({
                'box': [x, y, w, h],
                'name': f"{person_name} ({person_reg_no})",
                'confidence': f"{confidence*100:.2f}%"
            })
    except Exception as e:
        print(f"ERROR: Exception during face recognition: {e}")
        # Optionally return an error status to the frontend
        # return jsonify({'results': [], 'error': f'Recognition error: {e}'})

    return jsonify({'results': detected_faces})

# --- Main Execution ---
if __name__ == '__main__':
    init_db()
    load_user_registry()
    load_recognition_model()
    os.makedirs(RAW_IMAGE_DIR, exist_ok=True)
    os.makedirs(PROCESSED_IMAGE_DIR, exist_ok=True)
    # use_reloader=False prevents Flask from restarting during training
    app.run(debug=True, threaded=True, use_reloader=False)