# Phone Pose Estimation using OpenCV

This project contains two Python scripts designed for estimating the pose of a camera relative to its environment by analyzing images captured from an IP camera or a smartphone camera. Using SIFT (Scale-Invariant Feature Transform) for feature detection and FLANN-based matching, the scripts compute relative camera movement, which can be useful for tasks like augmented reality, camera tracking, or robot localization.

## Files in this Repository

1. **pose_estimator.py**: This file contains a class for estimating the relative pose between two consecutive images. It matches feature points between images and calculates the rotation and translation matrices.
2. **phone_pose.py**: This script implements the image capture pipeline, pose estimation, and visualization using the `pose_estimator` class. It captures images from a network camera (e.g., a phone's camera via an IP stream), performs feature matching, and computes the translation and rotation of the camera between frames.

## How it Works

### `pose_estimator.py`
The `pose_estimator` class takes camera intrinsic parameters, captures images, and estimates the camera's pose using the following steps:

1. **Feature Detection**: SIFT is used to detect features in two consecutive images.
2. **Feature Matching**: Matches are found between the features of two images using FLANN (Fast Library for Approximate Nearest Neighbors).
3. **Pose Estimation**: Using the matches, the essential matrix is computed, and the camera’s rotation (R) and translation (t) relative to the previous frame are estimated.
4. **Global Position Tracking**: The script maintains the camera's global rotation and translation using the incremental estimates from consecutive frames.

### `phone_pose.py`
This script ties together the pose estimation and the image capturing process:

1. **Input**: 
   - IP address of the camera (typically a smartphone acting as a network camera).
   - Camera intrinsic parameters: focal lengths (`fx`, `fy`) and optical center coordinates (`cx`, `cy`).
2. **Image Capture**: Images are captured from an IP stream using OpenCV’s `VideoCapture` function in a separate thread.
3. **Pose Estimation**: The `pose_estimator` class is used to process each captured frame and estimate the camera’s relative pose between frames.
4. **Visualization**: The matching features and global camera movement are visualized in a window.

## Requirements

- Python 3.x
- OpenCV (`opencv-python`)
- NumPy (`numpy`)
- A network camera feed (e.g., IP webcam on your phone), you can install [IP Webcam](https://play.google.com/store/apps/details?id=com.pas.webcam&hl=en_US) on your Android smartphone

You can install the required dependencies using:

```bash
pip install numpy opencv-python
