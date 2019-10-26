import cv2
from keras.applications.imagenet_utils import preprocess_input
import numpy as np


face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')


def preprocess_photo(img_name, dir_path='input_photo/'):
    image_path = dir_path + img_name
    img = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        img_gray,
        scaleFactor=1.1,
        minNeighbors=10,
        minSize=(10, 10)
    )

    faces_img = []
    changed_imgs = []
    for (x, y, w, h) in faces:
        new_face = cv2.resize(img[y: y + h, x: x + w], dsize=(224, 224))
        faces_img.append(cv2.cvtColor(new_face, cv2.COLOR_BGR2RGB))
        changed_imgs.append(preprocess_input(np.expand_dims(new_face, axis=0)))

    return faces_img, changed_imgs


def get_img(img_name, dir_path='input_photo/'):
    faces = preprocess_photo(img_name, dir_path=dir_path)
    print("Лиц обнаружено: {}".format(len(faces[0])))

    return faces
