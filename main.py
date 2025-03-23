import cv2
import numpy as np
import streamlit as st
import os

# path of input images to detect in frames
folder_path = 'Images'
if not os.path.exists(folder_path):   # if path does not exist, create it
    os.makedirs(folder_path)

# creating all images path and storing these images in a list 
model_path_list = os.listdir(folder_path)
model_path_list = [os.path.join(folder_path, path) for path in model_path_list]

image_mode_list = []
for path in model_path_list: 
    image_mode_list.append(path)

print("Number of different faces stored as input for future detection",len(image_mode_list))


# Capturing video from webcam
cap  = cv2.VideoCapture(0)

st.title("Face Attendance System")

frame_placeholder = st.empty()

stop_button = st.button("Stop Detection")

while cap.isOpened() and not stop_button:
    ret, frame = cap.read()
    if not ret:
        st.write("Failed to capture image")
        break

    # Convert the frame to RGB, cv2 uses BGR by default and Streamlit uses RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Display the frame in Streamlit
    frame_placeholder.image(rgb_frame, channels="RGB")

    if cv2.waitKey(1) & 0xFF == ord('q') or stop_button:
        break


# Release the video capture object
cap.release()   
cv2.destroyAllWindows()