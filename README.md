# Steel Surface Scratch and Dent Detection

This project implements an automated defect detection system for steel surfaces using deep learning. Built with YOLOv8, the system identifies scratches and dents in images or real-time input, improving inspection accuracy and reducing manual effort in industrial environments.

---

## Components Used

* Python
* YOLOv8 (Ultralytics)
* OpenCV
* NumPy
* Trained Dataset (Steel Surface Images)

---

## Working Principle

The system uses a trained YOLOv8 object detection model to analyze input images or video frames. It processes the data and detects defects based on learned patterns:

* Input image or camera feed is captured
* The YOLOv8 model processes the frame
* Defects such as scratches and dents are detected
* Bounding boxes are drawn around detected defects
* Results are displayed in real time

---

## Features

* Automated defect detection
* Real-time image processing
* High accuracy using deep learning
* Supports both image and camera input
* Visual output with bounding boxes

---

## Future Improvements

* Improve model accuracy with larger datasets
* Deploy on edge devices for real-time industrial use
* Integrate IoT-based monitoring system
* Optimize performance for faster inference
  
---

## Author

Hari Prasanna
B.Tech Robotics Engineering

---
