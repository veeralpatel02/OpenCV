# Smart Object Detection System

Welcome to the Smart Object Detection System! This system utilizes computer vision techniques to detect various objects and patterns in a live video feed. It employs FastAPI for backend processing and streaming of video, along with OpenCV for computer vision tasks.

## Features:

- **Multiple Object Detection**: Detects various objects, including red objects, light-blue objects, circular objects, faces, and QR codes in real-time video streams.
- **Interactive Web Interface**: Provides a user-friendly web interface to view live video streams and detected objects.
- **PID Control for Object Tracking**: Implements PID (Proportional-Integral-Derivative) control to track and follow detected objects, ensuring precise and efficient object tracking.
- **QR Code Detection**: Utilizes QR code detection to identify and extract QR code data from the video stream.
- **Object Recognition**: Incorporates YOLO (You Only Look Once) deep learning model for object recognition, enabling accurate detection of various objects from the COCO dataset.

## Requirements:

- Python 3.7 or higher
- OpenCV
- FastAPI
- Uvicorn

## Getting Started:

1. Clone the repository: `git clone https://github.com/yourusername/smart-object-detection.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the application: `python main.py`
4. Open your web browser and navigate to `http://localhost:8000` to view the live video feed and detected objects.

## Code Overview:

The main components of the code include:

- **Object Detection Functions**: Functions for detecting various objects like red objects, light-blue objects, circular objects, faces, and QR codes using OpenCV.
- **Streaming Functions**: Functions for streaming the processed video frames using FastAPI's `StreamingResponse`.
- **Web Interface**: FastAPI routes for serving HTML templates and handling video streaming requests.
- **Main Function**: Entry point of the application, where the FastAPI server is started using Uvicorn.

## Note:

Ensure that your camera is connected and properly configured before running the application. You may need to adjust camera settings or paths to the pre-trained models based on your setup.

Feel free to explore and customize the code according to your requirements!

