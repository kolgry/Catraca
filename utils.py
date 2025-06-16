# utils.py
import cv2
import numpy as np
import os
import re

def validate_numeric_input(P):
    if P.isdigit() or P == "":
        return True
    else:
        return False

def resize_video(width, height, max_width = 600):
  if (width > max_width):
    proportion = width / height
    video_width = max_width
    video_height = int(video_width / proportion)
  else:
    video_width = width
    video_height = height
  return video_width, video_height

def parse_name(name):
    name = re.sub(r"[^\w\s]", '', name)
    name = re.sub(r"\s+", '_', name)
    return name

def create_folders(final_path, final_path_full):
    if not os.path.exists(final_path):
        os.makedirs(final_path)
    if not os.path.exists(final_path_full):
        os.makedirs(final_path_full)

def detect_face(face_detector, orig_frame):
    frame = orig_frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.1, 5)

    face_roi = None
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (0, 255, 255), 2)
        face_roi = orig_frame[y:y + h, x:x + w]
        face_roi = cv2.resize(face_roi, (140, 140))
    return face_roi, frame

def detect_face_ssd(network, orig_frame, show_conf=True, conf_min=0.7):
    frame = orig_frame.copy()
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 117.0, 123.0))
    network.setInput(blob)
    detections = network.forward()

    face_roi = None
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_min:
            bbox = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (start_x, start_y, end_x, end_y) = bbox.astype("int")

            if (start_x < 0 or start_y < 0 or end_x > w or end_y > h):
                continue

            face_roi = orig_frame[start_y:end_y, start_x:end_x]
            face_roi = cv2.resize(face_roi, (90, 120))
            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
            if show_conf:
                text_conf = "{:.2f}%".format(confidence * 100)
                cv2.putText(frame, text_conf, (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return face_roi, frame