# Face Mask Detection System
<img src="https://news.cgtn.com/news/77497a4e7a457a4e3241444d34636a4e3359444f31457a6333566d54/img/d87b2bb0ca8e47dcbff030e6d644f7de/d87b2bb0ca8e47dcbff030e6d644f7de.jpg">

## Overview
This project is a Face Mask Detection System developed using Streamlit, OpenCV, Keras, and TensorFlow. The system can detect whether individuals are wearing a mask or not in images, videos, and real-time streams from a webcam or IP camera. It includes various functionalities for image and video analysis, and it uses a pre-trained deep learning model for mask detection.
## Objective
The objective of the Face Mask Detection System (FMDS) is to develop an automated solution that can accurately detect whether individuals are wearing face masks in real-time. The system utilizes machine learning models and computer vision techniques to analyze images, videos, and real-time streams from webcams or IP cameras. This application is aimed at promoting public health safety by assisting in the enforcement of mask-wearing protocols in public places, workplaces, and other environments.
## Features
  - **Image Face Mask Detection:** Upload an image to detect whether individuals are wearing masks.
  - **Video Face Mask Detection:** Upload a video to analyze mask-wearing status across frames.
  - **Web Camera Face Mask Detection:** Stream live video from your webcam and detect mask-wearing status in real-time.
  - **IP Camera Face Mask Detection:** Stream video from an IP camera and detect mask-wearing status.
## Technology Stack
  **Frontend:** Streamlit
  **Backend:** Python, Keras, OpenCV
  **Machine Learning Model:** Convolutional Neural Network (CNN) for mask detection
## Libraries:
  - streamlit
  - opencv-python
  - keras
  - numpy
  - PIL
## Installation
To get a copy of the project up and running locally, follow these steps:

**Prerequisites**
Python 3.x installed
Virtual environment (optional but recommended)
  1. **Clone the Repository:**

    ```bash
    git clone https://github.com/yourusername/face-mask-detection.git
    ```
  2. **Navigate to the Project Directory**

     ```bash
      cd face-mask-detection
     ```
  3. **Create a Virtual Environment (Optional but recommended)**

      ```bash
        python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
      ```
  4. **Install Required Packages**

    Install the required packages using pip:
    ```bash
      pip install -r requirements.txt
    ```
  5. **Download Pre-trained Models**
  Make sure you have the face.xml and mask.h5 files in the project directory. You can find these files:

  - Face Cascade XML[Download Here](face.xml)
  - Mask Detection Model[Download Here](mask.h5)
## Usage
  1. **Run the Streamlit App**

      ```bash
      streamlit run app.py
      ```
  2. **Navigate to the Localhost URL**

    Open your web browser and go to http://localhost:8501 to access the Face Mask Detection System.

## Project Structure
  ```bash
  
  Facemask-Detection-System/
  │
  ├── face.xml                       # Pre-trained face detection model (Haar Cascade)
  ├── mask.h5                        # Pre-trained mask detection model (Keras)
  ├── app.py                         # Main Streamlit application file
  ├── requirements.txt               # Required dependencies
  └── README.md                      # Project documentation

  ```

