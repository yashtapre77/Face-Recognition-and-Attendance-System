import cv2
import numpy as np
import streamlit as st
import os
import pickle 
import face_recognition

# Load the face encodings from the pickle file
file = open("face_encodings.pickle",'rb')
face_encodings_dict = pickle.load(file)
file.close()

def compare_faces(frame, face_encodings ):
    i = 0
    matches = None
    face_dis = None
    # Compare the face encodings with the known faces
    for encode_face in face_encodings:
        matches = face_recognition.compare_faces(list(face_encodings_dict.values()), encode_face)
        face_dis = face_recognition.face_distance(list(face_encodings_dict.values()), encode_face)

        print(i," Matches:", matches)
        print(i," Face distances:", face_dis)
        i += 1


    return matches, face_dis


def encode_video_faces(frame):
    # Convert the frame to RGB and resize it for faster processing to 1/4 size
    rgb_frame = cv2.resize(frame, (0, 0), None, 0.25, 0.25)

    # Find all face locations and encodings in the frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # for encode_face, face_loc in zip(face_encodings, face_locations):
    #     # Compare the face encodings with the known faces
    #     matches = face_recognition.compare_faces(list(face_encodings_dict.values()), encode_face)
    #     face_dis = face_recognition.face_distance(list(face_encodings_dict.values()), encode_face)
    #     name = "Unknown"

    #     # If a match is found, get the name of the matched face
    #     if True in matches:
    #         first_match_index = matches.index(True)
    #         name = list(face_encodings_dict.keys())[first_match_index]

    #     # Draw a rectangle around the face and put the name on it
    #     top, right, bottom, left = face_loc
    #     top *= 4
    #     right *= 4
    #     bottom *= 4
    #     left *= 4
    # print(face_di)
    # for i in range(len(face_locations)):
    #     top, right, bottom, left = face_locations[i]
    #     cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
    # #     cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame, face_locations, face_encodings



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
    new_frame, face_locations, face_encodings = encode_video_faces(rgb_frame)
    matches, face_dis = compare_faces(rgb_frame, face_encodings)


    # Display the frame in Streamlit
    frame_placeholder.image(new_frame, channels="RGB")

    if cv2.waitKey(1) & 0xFF == ord('q') or stop_button:
        break


# Release the video capture object
cap.release()   
cv2.destroyAllWindows()




