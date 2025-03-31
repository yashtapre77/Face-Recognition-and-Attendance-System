import cv2
import os
import face_recognition
import pickle

def encode_faces(directory_path): 
    # path of input images to detect in frames
    folder_path = directory_path
    if not os.path.exists(folder_path):   # if path does not exist, create it
        os.makedirs(folder_path)

    # creating all images path and storing these images in a list 
    images_name_list = os.listdir(folder_path)
    images_path_dict = {}
    face_encodings_dict = {}

    for path in images_name_list:
        face_id  =  path.split('.')[0]
        full_path = os.path.join(folder_path, path)
        images_path_dict[face_id] = full_path
        img = cv2.imread(full_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_encodings_dict[face_id] = face_recognition.face_encodings(img)[0]

    return face_encodings_dict 


ret = encode_faces('Images')
file = open('face_encodings.pickle', 'wb')
pickle.dump(ret, file)
file.close()
# print("Number of different faces stored as input for future detection", len(ret))
# print("Face encodings dictionary:", ret)
