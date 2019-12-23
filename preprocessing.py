import cv2
from keras.applications.imagenet_utils import preprocess_input
import numpy as np


face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')


def preprocess_photo(img_name, dir_path='input_photo/', add_noize=False):
    image_path = dir_path + img_name
    img = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        img_gray,
        scaleFactor=1.1,
        minNeighbors=30,
        minSize=(10, 10)
    )

    faces_img = []
    changed_imgs = []
    for (x, y, w, h) in faces:
        img2 = img.copy()
        if add_noize:
            mean = 20
            var = 1000
            sigma = var ** 0.5
            gaussian = np.random.normal(mean, sigma, (h, w))
            img2[y: y + h, x: x + w, 0] = img[y: y + h, x: x + w, 0] + gaussian
            img2[y: y + h, x: x + w, 1] = img[y: y + h, x: x + w, 1] + gaussian
            img2[y: y + h, x: x + w, 2] = img[y: y + h, x: x + w, 2] + gaussian

        new_face = cv2.resize(img2[y: y + h, x: x + w], dsize=(224, 224))
        # faces_img.append(cv2.rectangle(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB), (x, y), (x + w, y + h), (255, 0, 0), thickness=5))
        faces_img.append(cv2.rectangle(img2, (x, y), (x + w, y + h), (255, 0, 0), thickness=5))
        changed_imgs.append(preprocess_input(np.expand_dims(new_face, axis=0)))

    return faces_img, changed_imgs


def get_img(img_name, dir_path='input_photo/', add_noize=False):
    faces = preprocess_photo(img_name, dir_path=dir_path, add_noize=add_noize)
    print("Лиц обнаружено: {}".format(len(faces[0])))

    return faces
