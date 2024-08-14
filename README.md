# Face Mask Detection System
## Overview
This project is a Face Mask Detection System developed using Streamlit, OpenCV, Keras, and TensorFlow. The system can detect whether individuals are wearing a mask or not in images, videos, and real-time streams from a webcam or IP camera. It includes various functionalities for image and video analysis, and it uses a pre-trained deep learning model for mask detection.

## Features
  - **Image Face Mask Detection:** Upload an image to detect whether individuals are wearing masks.
  - **Video Face Mask Detection:** Upload a video to analyze mask-wearing status across frames.
  - **Web Camera Face Mask Detection:** Stream live video from your webcam and detect mask-wearing status in real-time.
  - **IP Camera Face Mask Detection:** Stream video from an IP camera and detect mask-wearing status.
## Installation
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

How It Works
Image Face Mask Detection:

Upload an image.
The system detects faces and checks if masks are being worn.
Results are displayed with bounding boxes and labels.
Video Face Mask Detection:

Upload a video.
The system processes each frame to detect mask-wearing status.
Results are displayed with bounding boxes and labels for each frame.
Web Camera Face Mask Detection:

Click "Start Camera" to begin live video feed from your webcam.
The system detects mask-wearing status in real-time.
IP Camera Face Mask Detection:

Enter the URL of your IP camera and click "Start Camera."
The system detects mask-wearing status in the video feed from the IP camera.
Contributing
Feel free to open issues and submit pull requests if you have suggestions or improvements.

License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
OpenCV for image processing.
Keras for deep learning model handling.
Streamlit for the web application interface.

Project Structure
bash
Copy code
Facemask-Detection-System/
│
├── face.xml                       # Pre-trained face detection model (Haar Cascade)
├── mask.h5                        # Pre-trained mask detection model (Keras)
├── app.py                         # Main Streamlit application file
├── requirements.txt               # Required dependencies
└── README.md                      # Project documentation



