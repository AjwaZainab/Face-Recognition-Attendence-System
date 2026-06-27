# **Facial Recognition Attendance System**

# **Overview**

This project implements a real-time, AI-powered attendance system using facial recognition. It automates the process of marking attendance by identifying enrolled users from a live video feed, making the process accurate and efficient.

## **Project Structure**

The repository is organized to separate the backend logic, machine learning assets, and frontend components for clarity and maintainability.

.  
├── dataset/                     \# Contains raw, un-processed images for each user (e.g., face captures)  
│   └── SP24-BAI-004/  
│   └── SP24-BAI-041/  
├── processed\_dataset/           \# Stores cropped and aligned faces derived from the raw dataset (used for training)  
│   └── SP24-BAI-004/  
│   └── SP24-BAI-041/  
├── static/                      \# Web assets (CSS and JavaScript)  
│   ├── css/  
│   │   └── style.css            \# Stylesheet for the web interface  
│   └── js/  
│       └── main.js              \# Frontend logic (e.g., webcam handling, API calls)  
├── templates/                   \# HTML templates for the Flask web application (Jinja)  
│   ├── dashboard.html           \# Main view after login, showing system summary  
│   ├── enroll.html              \# Interface for enrolling new users (capturing face images)  
│   ├── layout.html              \# Base template for consistent navigation and structure  
│   ├── live\_attendance.html     \# Real-time attendance logging via webcam  
│   ├── manage\_users.html        \# Admin page for viewing/editing user profiles  
│   └── view\_attendance.html     \# Page for viewing historical attendance records  
├── app.py                       \# Main Flask application file (handles routing, prediction, and database interaction)  
├── attendance.db                \# SQLite or similar database file for storing attendance logs  
├── attendance\_system.py         \# Secondary Python file (likely contains core ML/utility functions)  
├── facial\_recognition\_model.h5  \# The trained Keras/TensorFlow model (main ML asset)  
├── facial\_recognition\_model.tflite \# Optimized model for deployment (e.g., mobile or edge devices)  
├── label\_encoder.npy            \# NumPy array mapping numerical labels to user IDs  
└── user\_registry.json           \# JSON file storing metadata for enrolled users

## **Setup and Installation**

### **Prerequisites**

* Python 3.xx  
* Pip (Python package installer)  
* A webcam for live attendance and enrollment

### **Installation Steps**

1. **Clone the Repository:**  
   git clone \<repository-url\>  
   cd Facial-Recognition-Attendance-System

2. Install Dependencies:  
   This project requires common libraries like Flask, TensorFlow/Keras, OpenCV, and NumPy.  
   pip install flask tensorflow numpy opencv-python scikit-learn

   *(Note: You may need to create a requirements.txt file listing all dependencies for easier setup.)*  
3. Ensure Model Files are Present:  
   Verify that the following machine learning assets are in the root directory:  
   * facial\_recognition\_model.h5  
   * facial\_recognition\_model.tflite  
   * label\_encoder.npy

## **Usage**

### **1\. Data Preparation (Enrollment)**

1. **Collect Raw Data:** Place raw images for each user into dedicated subdirectories within the dataset/ folder.  
2. **Process Data:** Run the relevant script (attendance\_system.py or a function in app.py) to process the raw images (cropping, aligning, normalizing) and save the results into processed\_dataset/. This data is then used to train or update the model.

### **2\. Run the Application**

Start the Flask web server by running the main application file:

python app.py

The application will typically be available at http://127.0.0.1:5000/.

### **3\. Key Endpoints**

| Page (templates) | URL (Example) | Functionality |
| :---- | :---- | :---- |
| dashboard.html | / or /dashboard | Main administrative dashboard. |
| enroll.html | /enroll | Capture and register new user faces. |
| live\_attendance.html | /live | Real-time face detection and attendance logging. |
| manage\_users.html | /users | View, edit, or delete enrolled user profiles. |
| view\_attendance.html | /attendance | Browse and filter historical attendance data. |

## **Customization**

* **Styling:** Modify the appearance of the application by editing static/css/style.css.  
* **Frontend Logic:** Adjust webcam interactions, button handlers, or live video processing in static/js/main.js.  
* **Model Retraining:** After enrolling new users via enroll.html, you will need to re-run your training script (likely within attendance\_system.py) to update facial\_recognition\_model.h5 and label\_encoder.npy.