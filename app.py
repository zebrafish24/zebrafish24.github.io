import os
import cv2
import numpy as np
from scipy.spatial.distance import euclidean
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
from datetime import datetime
import pandas as pd

app = Flask(__name__)

# Configure file upload settings
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to check if the file is an allowed video format
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to calculate angle between two centroids
def calculate_angle_between_centroids(centroid1, centroid2):
    delta_x = centroid2[0] - centroid1[0]
    delta_y = centroid2[1] - centroid1[1]
    angle_rad = np.arctan2(delta_y, delta_x)
    angle_deg = np.degrees(angle_rad)
    angle_deg = abs(angle_deg) % 180
    return angle_deg

# Main route to display the upload page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle video upload and processing
@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file part'})
    
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if video_file and allowed_file(video_file.filename):
        filename = secure_filename(video_file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        video_file.save(video_path)

        # Process the video with the quivering detection logic
        result = process_video(video_path)

        # Generate the result file (text or Excel)
        result_file = generate_result_file(result)

        # Return JSON with file path and result data
        return jsonify({
            'file_path': result_file,
            'quiver_count': result['quiver_count'],
            'timestamps': result['timestamps']
        })
    else:
        return jsonify({'error': 'File format not allowed'})

# Function to process the video and detect quivering
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    backSub = cv2.createBackgroundSubtractorMOG2(history=1500, varThreshold=50, detectShadows=False)

    distance_threshold = 5
    angle_threshold = 1
    quiver_frame_count = 0
    quiver_persistence_threshold = 10
    quiver_time_threshold = 3

    prev_time = 0
    quiver_start_time = 0

    quiver_timestamps = []  # List to store timestamps
    quiver_count = 0  # Variable to track number of quivers detected

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_time = cv2.getTickCount() / cv2.getTickFrequency()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
        fg_mask = backSub.apply(blurred_frame)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        clean_fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        clean_fg_mask = cv2.morphologyEx(clean_fg_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(clean_fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        current_centroids = []

        for contour in contours:
            if cv2.contourArea(contour) > 500:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    current_centroids.append((cx, cy))

        if len(current_centroids) == 2:
            centroid1, centroid2 = current_centroids[:2]
            distance = euclidean(centroid1, centroid2)
            angle = calculate_angle_between_centroids(centroid1, centroid2)

            if distance < distance_threshold and angle < angle_threshold:
                if quiver_frame_count == 0:
                    quiver_start_time = current_time
                quiver_frame_count += 1
                if current_time - quiver_start_time >= quiver_time_threshold:
                    quiver_timestamps.append(datetime.now().strftime('%H:%M:%S'))  # Save timestamp
                    quiver_count += 1
            else:
                pass

        elif len(current_centroids) == 1:
            quiver_frame_count += 1
            if quiver_frame_count >= quiver_persistence_threshold:
                quiver_timestamps.append(datetime.now().strftime('%H:%M:%S'))  # Save timestamp
                quiver_count += 1
        else:
            quiver_frame_count = 0

    cap.release()
    return {
        'timestamps': quiver_timestamps,
        'quiver_count': quiver_count
    }

# Function to generate the result file (text or Excel)
def generate_result_file(result):
    quiver_count = result['quiver_count']
    timestamps = result['timestamps']
    
    # Save as a text file
    text_file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'quiver_results.txt')
    with open(text_file_path, 'w') as f:
        f.write(f"Total Quivers Detected: {quiver_count}\n\n")
        f.write("Quivering Timestamps:\n")
        for timestamp in timestamps:
            f.write(f"{timestamp}\n")

    return text_file_path  # Return the text file path

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)