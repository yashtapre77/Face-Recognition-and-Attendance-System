import cv2
import numpy as np
import streamlit as st
import os
import pickle 
import face_recognition
import cvzone

# Load the face encodings from the pickle file
file = open("face_encodings.pickle",'rb')
face_encodings_dict = pickle.load(file)
file.close()

def locate_faces(frame, face_encodings , face_locations):
    matches = None
    face_dis = None
    # Compare the face encodings with the known faces
    for encode_face, face_locs in zip(face_encodings, face_locations): # for encode_face in face_encodings:
        matches = face_recognition.compare_faces(list(face_encodings_dict.values()), encode_face)
        face_dis = face_recognition.face_distance(list(face_encodings_dict.values()), encode_face)

        
        matches_index = np.argmin(face_dis)
        if  matches[matches_index]:
            name = list(face_encodings_dict.keys())[matches_index]
            top, right, bottom, left = face_locs
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            cvzone.cornerRect(frame, (left, top, right - left, bottom - top), rt=0)
            cv2.putText(frame, name, (left, top - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)


    # return  matches, face_dis
    return frame


def encode_video_faces(frame):
    # Convert the frame to RGB and resize it for faster processing to 1/4 size
    frame = cv2.resize(frame, (0, 0), None, 0.25, 0.25)

    # Find all face locations and encodings in the frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)


    return face_locations, face_encodings



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
    face_locations, face_encodings = encode_video_faces(rgb_frame)
    new_frame= locate_faces(rgb_frame, face_encodings, face_locations)


    # Display the frame in Streamlit
    frame_placeholder.image(new_frame, channels="RGB")

    if cv2.waitKey(1) & 0xFF == ord('q') or stop_button:
        break


# Release the video capture object
cap.release()   
cv2.destroyAllWindows()




