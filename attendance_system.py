import os
import sys
import time
import cv2
import numpy as np
import sqlite3
from datetime import datetime
import matplotlib.pyplot as plt

# Suppress TensorFlow logging messages for a cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# Conditional imports based on availability
try:
    from mtcnn import MTCNN
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
except ImportError as e:
    print(f"Error: A required library is missing. {e}")
    print("Please install all required libraries by running:")
    print("pip install tensorflow opencv-python mtcnn scikit-learn matplotlib")
    sys.exit(1)

# --- System Configuration ---
RAW_IMAGE_DIR = 'dataset/'
PROCESSED_IMAGE_DIR = 'processed_dataset/'
DB_FILE = 'attendance.db'
MODEL_PATH_H5 = 'facial_recognition_model.h5'
MODEL_PATH_TFLITE = 'facial_recognition_model.tflite'
LABEL_ENCODER_PATH = 'label_encoder.npy'

# Image properties
IMG_SIZE = (160, 160)
IMAGES_TO_CAPTURE = 30

# Model Training properties
BATCH_SIZE = 32
EPOCHS = 50


# --- 1. Add New User ---
def add_new_user():
    """Captures and saves images for a new user."""
    print("\n--- Starting New User Enrollment ---")
    person_name = input("Enter the name of the new person: ").strip().replace(" ", "_")
    if not person_name:
        print("Name cannot be empty.")
        return

    person_folder = os.path.join(RAW_IMAGE_DIR, person_name)
    if not os.path.exists(person_folder):
        os.makedirs(person_folder)
        print(f"Directory created for {person_name} at {person_folder}")
    else:
        print(f"Directory for {person_name} already exists.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print(f"\nStarting image capture for {person_name}. Look at the camera.")
    print(f"Capturing {IMAGES_TO_CAPTURE} images...")

    for i in range(IMAGES_TO_CAPTURE):
        for j in range(1, 0, -1):
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame.")
                continue

            # Display countdown on the frame
            text = f"Capturing image {i + 1}/{IMAGES_TO_CAPTURE} in {j}..."
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow('Add New User', frame)
            cv2.waitKey(500)

        ret, frame = cap.read()
        if not ret:
            print(f"Failed to capture image {i + 1}.")
            continue

        image_path = os.path.join(person_folder, f"{person_name}_{int(time.time())}_{i}.jpg")
        cv2.imwrite(image_path, frame)
        print(f"Saved {image_path}")

        # Show a "Saved!" message
        cv2.putText(frame, "SAVED!", (150, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        cv2.imshow('Add New User', frame)
        cv2.waitKey(500)

    cap.release()
    cv2.destroyAllWindows()
    print("\n--- Enrollment Finished ---")
    print("IMPORTANT: You must now run 'Train Model' to include the new user.")


# --- 2. Train Model ---
def prepare_dataset():
    """Detects, crops, and resizes faces from the raw dataset."""
    print("\n--- Starting Dataset Preparation ---")
    if not os.path.exists(RAW_IMAGE_DIR) or not os.listdir(RAW_IMAGE_DIR):
        print(f"Error: Raw dataset directory '{RAW_IMAGE_DIR}' is empty or not found.")
        print("Please add users first using option 1.")
        return False

    if not os.path.exists(PROCESSED_IMAGE_DIR):
        os.makedirs(PROCESSED_IMAGE_DIR)

    detector = MTCNN()
    total_processed = 0

    for person_name in os.listdir(RAW_IMAGE_DIR):
        person_folder_in = os.path.join(RAW_IMAGE_DIR, person_name)
        person_folder_out = os.path.join(PROCESSED_IMAGE_DIR, person_name)

        if not os.path.isdir(person_folder_in):
            continue

        if not os.path.exists(person_folder_out):
            os.makedirs(person_folder_out)

        image_count = 0
        for i, image_name in enumerate(os.listdir(person_folder_in)):
            image_path = os.path.join(person_folder_in, image_name)
            try:
                img = cv2.imread(image_path)
                if img is None:
                    print(f"Warning: Could not read image {image_path}")
                    continue

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = detector.detect_faces(img_rgb)

                if results:
                    x, y, w, h = results[0]['box']
                    x1, y1 = max(0, x), max(0, y)
                    x2, y2 = min(img.shape[1], x + w), min(img.shape[0], y + h)
                    face = img[y1:y2, x1:x2]

                    if face.size == 0:
                        print(f"Warning: Detected face in {image_path} is empty. Skipping.")
                        continue

                    resized_face = cv2.resize(face, IMG_SIZE)
                    save_path = os.path.join(person_folder_out, f"{person_name}_{i}.jpg")
                    cv2.imwrite(save_path, resized_face)
                    image_count += 1
                else:
                    print(f"Warning: No face detected in {image_path}")
            except Exception as e:
                print(f"Error processing {image_path}: {e}")

        print(f"Processed {image_count} images for {person_name}.")
        total_processed += image_count

    if total_processed == 0:
        print("Error: No faces were successfully processed. Training cannot continue.")
        return False

    print("--- Dataset Preparation Finished ---\n")
    return True


def train_model():
    """Coordinates the full training pipeline."""
    if not prepare_dataset():
        return

    print("--- Loading and Preprocessing Data for Training ---")
    images, labels = [], []

    for person_name in os.listdir(PROCESSED_IMAGE_DIR):
        person_folder = os.path.join(PROCESSED_IMAGE_DIR, person_name)
        if not os.path.isdir(person_folder):
            continue
        for image_name in os.listdir(person_folder):
            img_path = os.path.join(person_folder, image_name)
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img.astype('float32') / 255.0)
                labels.append(person_name)

    if len(images) < 2:
        print("Error: Not enough data to train. Need at least two processed images.")
        return

    images = np.array(images)
    labels = np.array(labels)

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    num_classes = len(np.unique(encoded_labels))

    if num_classes < 2:
        print("Error: Training requires at least 2 different people. Please add more users.")
        return

    categorical_labels = tf.keras.utils.to_categorical(encoded_labels, num_classes)
    np.save(LABEL_ENCODER_PATH, label_encoder.classes_)
    print(f"\nSaved label mapping: {label_encoder.classes_}")

    X_train, X_test, y_train, y_test = train_test_split(
        images, categorical_labels, test_size=0.2, random_state=42, stratify=categorical_labels
    )
    print("--- Data Loading Finished ---\n")

    print("--- Building CNN Model ---")
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    datagen = ImageDataGenerator(
        rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
        shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest'
    )
    datagen.fit(X_train)

    print("\n--- Starting Model Training ---")
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        steps_per_epoch=max(1, len(X_train) // BATCH_SIZE),
        validation_data=(X_test, y_test),
        epochs=EPOCHS
    )
    print("--- Model Training Finished ---\n")

    print("--- Evaluating Model ---")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f'Test Accuracy: {accuracy * 100:.2f}%')

    # Save training history plot
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.legend()
    plt.savefig('training_history.png')
    print("Saved training history plot to 'training_history.png'")

    print("\n--- Saving Models ---")
    model.save(MODEL_PATH_H5)
    print(f"Keras model saved as '{MODEL_PATH_H5}'")

    # Convert and save TFLite model
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        with open(MODEL_PATH_TFLITE, 'wb') as f:
            f.write(tflite_model)
        print(f"TFLite model saved as '{MODEL_PATH_TFLITE}'")
    except Exception as e:
        print(f"Error converting to TFLite model: {e}")


# --- 3. Run Attendance System ---
def run_attendance():
    """Initializes webcam and runs real-time recognition and attendance logging."""
    print("\n--- Starting Live Attendance System ---")

    try:
        model = load_model(MODEL_PATH_H5)
        labels = np.load(LABEL_ENCODER_PATH, allow_pickle=True)
    except Exception as e:
        print(f"Error: Could not load model or labels. {e}")
        print("Please train the model first using option 2.")
        return

    detector = MTCNN()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    today_date = datetime.now().strftime('%Y-%m-%d')
    marked_today = set()
    # Load names of people already marked today
    cursor.execute("SELECT name FROM attendance WHERE date = ?", (today_date,))
    for row in cursor.fetchall():
        marked_today.add(row[0])

    print("Webcam started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.detect_faces(img_rgb)

        for result in results:
            x, y, w, h = result['box']
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)

            face = frame[y1:y2, x1:x2]
            if face.size == 0: continue

            resized_face = cv2.resize(face, IMG_SIZE)
            face_array = resized_face.astype('float32') / 255.0
            face_array = np.expand_dims(face_array, axis=0)

            predictions = model.predict(face_array, verbose=0)
            confidence = np.max(predictions[0])

            if confidence > 0.70:  # Confidence threshold
                class_index = np.argmax(predictions[0])
                person_name = labels[class_index]

                if person_name not in marked_today:
                    current_time = datetime.now().strftime('%H:%M:%S')
                    cursor.execute("INSERT INTO attendance (name, date, time) VALUES (?, ?, ?)",
                                   (person_name, today_date, current_time))
                    conn.commit()
                    marked_today.add(person_name)
                    print(f"Attendance marked for {person_name} at {current_time}")

                text = f"{person_name} ({confidence * 100:.2f}%)"
                color = (0, 255, 0)  # Green for recognized
            else:
                text = "Unknown"
                color = (0, 0, 255)  # Red for unknown

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow('Facial Recognition Attendance - Press Q to Quit', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    conn.close()
    print("--- Attendance System Finished ---")


# --- 4. Check Attendance ---
def check_attendance():
    """Provides options to view attendance records from the database."""
    while True:
        print("\n--- Check Attendance Menu ---")
        print("1. View Today's Attendance")
        print("2. View Attendance by Date")
        print("3. View All Attendance for a Person")
        print("4. Back to Main Menu")
        choice = input("Enter your choice: ").strip()

        if choice == '1':
            view_by_date(datetime.now().strftime('%Y-%m-%d'))
        elif choice == '2':
            date_str = input("Enter date (YYYY-MM-DD): ").strip()
            try:
                datetime.strptime(date_str, '%Y-%m-%d')
                view_by_date(date_str)
            except ValueError:
                print("Invalid date format. Please use YYYY-MM-DD.")
        elif choice == '3':
            name = input("Enter person's name: ").strip().replace(" ", "_")
            view_by_name(name)
        elif choice == '4':
            break
        else:
            print("Invalid choice. Please try again.")


def view_by_date(date_str):
    """Displays attendance records for a specific date."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT name, time FROM attendance WHERE date = ? ORDER BY time", (date_str,))
    records = cursor.fetchall()
    conn.close()

    if records:
        print(f"\n--- Attendance for {date_str} ---")
        print(f"{'Name':<20} | {'Time'}")
        print("-" * 30)
        for name, time in records:
            print(f"{name:<20} | {time}")
    else:
        print(f"No attendance records found for {date_str}.")


def view_by_name(name):
    """Displays all attendance records for a specific person."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT date, time FROM attendance WHERE name = ? ORDER BY date DESC, time", (name,))
    records = cursor.fetchall()
    conn.close()

    if records:
        print(f"\n--- Attendance Records for {name} ---")
        print(f"{'Date':<15} | {'Time'}")
        print("-" * 25)
        for date, time in records:
            print(f"{date:<15} | {time}")
    else:
        print(f"No attendance records found for {name}.")


# --- Main Application Logic ---
def init_db():
    """Initializes the SQLite database and creates the attendance table if it doesn't exist."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            UNIQUE(name, date)
        )
    ''')
    conn.commit()
    conn.close()


def main():
    """Main function to display the menu and run the application."""
    init_db()
    while True:
        print("\n\n===== Facial Recognition Attendance System =====")
        print("1. Add New User")
        print("2. Train Model")
        print("3. Run Attendance System")
        print("4. Check Attendance")
        print("5. Exit")
        choice = input("Enter your choice [1-5]: ").strip()

        if choice == '1':
            add_new_user()
        elif choice == '2':
            train_model()
        elif choice == '3':
            run_attendance()
        elif choice == '4':
            check_attendance()
        elif choice == '5':
            print("Exiting the system. Goodbye!")
            break
        else:
            print("Invalid choice, please enter a number between 1 and 5.")


if __name__ == '__main__':
    main()