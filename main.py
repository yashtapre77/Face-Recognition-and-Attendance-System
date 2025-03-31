import cv2
import numpy as np
import streamlit as st
import os
import pickle 
import face_recognition

# path of input images to detect in frames
folder_path = 'Images'
if not os.path.exists(folder_path):   # if path does not exist, create it
    os.makedirs(folder_path)

# creating all images path and storing these images in a list, along with their ids
images_name_list = os.listdir(folder_path)
images_path_dict = {}

for path in images_name_list: 
    face_id  =  path.split('.')[0]
    full_path = os.path.join(folder_path, path)
    images_path_dict[face_id] = full_path


# load encoded image data from pickle file
file = open("face_encodings.pickle",'rb')
face_encodings_dict = pickle.load(file)
file.close()

print("Face encodings dictionary:", face_encodings_dict)
print("Number of different faces stored as input for future detection",len(images_path_dict))


# Capturing video from webcam
cap  = cv2.VideoCapture(0)

# setting streamlit page 
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




def check_face_encodings(frame):
    # Convert the frame to RGB and resize it for faster processing to 1/4 size
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame = cv2.resize(rgb_frame, (0, 0), None, 0.25, 0.25)

    # Find all face locations and encodings in the frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    return face_locations, face_encodings