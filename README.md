# Face Recognition Attendance System

## Overview

The Face Recognition Attendance System is a deep learning-based web application that automates attendance marking using facial recognition. It detects and recognizes registered users through a webcam and records their attendance in a database, eliminating the need for manual attendance.

---

## Features

* Real-time face detection and recognition
* Automatic attendance marking
* User registration
* Attendance records stored in a database
* Web-based interface built with Flask
* Deep learning model for accurate face recognition
* Lightweight TensorFlow Lite model for faster inference

---

## Technologies Used

* Python
* Flask
* OpenCV
* TensorFlow / TensorFlow Lite
* NumPy
* HTML, CSS, JavaScript
* SQLite

---

## Project Structure

```text
Face-Recognition-Attendance-System/
│
├── static/
├── templates/
├── app.py
├── attendance_system.py
├── attendance.db
├── user_registry.json
├── facial_recognition_model.tflite
├── label_encoder.npy
├── README.md
└── requirements.txt
```

---

## Installation

1. Clone the repository

```bash
git clone https://github.com/your-username/Face-Recognition-Attendance-System.git
```

2. Navigate to the project folder

```bash
cd Face-Recognition-Attendance-System
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Run the application

```bash
python app.py
```

5. Open your browser and visit

```
http://127.0.0.1:5000
```

---

## Dataset

The original training dataset is **not included** in this repository because it contains private personal images.

To use this project:

* Create your own `dataset/` folder.
* Add images of authorized users.
* Retrain the face recognition model using your dataset.

---

## Model Files

The original trained model is not included because it was trained on private images.

You can train a new model using your own dataset or replace the model files with your own trained versions.

---

## Future Improvements

* Multiple face recognition
* Email notifications
* Cloud database integration
* User authentication
* Attendance analytics dashboard
* Mobile application support

---

## License

This project is developed for educational and learning purposes.

---

## Author

**Ajwa Zainab**

Bachelor of Artificial Intelligence

Interested in Artificial Intelligence, Machine Learning, Computer Vision, and Deep Learning.
