import cv2
from keras.applications.imagenet_utils import preprocess_input
from foolbox.attacks import GradientSignAttack
import numpy as np
import matplotlib.pyplot as plt
import metrics
import tensorflow as tf
import foolbox
tf.compat.v1.disable_eager_execution()
tf.compat.v1.experimental.output_all_intermediates(True)
from foolbox.criteria import Misclassification


face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')


def preprocess_photo(img_name, dir_path='input_photo/', add_noize=False, db_model=None):
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
            foolbox_model = foolbox.models.KerasModel(model=db_model[1], bounds=(0, 255))
            tmp_face = np.expand_dims(cv2.resize(img2[y: y + h, x: x + w], dsize=(224, 224)), axis=0)
            attack = foolbox.attacks.FGSM(foolbox_model)  # default is Misclassification
            new_face = attack(tmp_face, labels=np.ndarray(1))
            print('real class: {}'.format(db_model[1].predict(tmp_face)))
            print('new class: {}'.format(db_model[1].predict(new_face)))
            # fig, imgs = plt.subplots(1, 2, num='changed')
            # for ax in imgs:
            #     ax.set_xticks([])
            #     ax.set_yticks([])
            # imgs[0].imshow(img2[y: y + h, x: x + w])
            # imgs[0].set_title('old')
            # imgs[1].imshow(new_face[0])
            # imgs[1].set_title('new')
            # plt.show()
            faces_img.append(tmp_face[0])
        else:
            new_face = np.expand_dims(cv2.resize(img2[y: y + h, x: x + w], dsize=(224, 224)), axis=0)
        # faces_img.append(cv2.rectangle(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB), (x, y), (x + w, y + h), (255, 0, 0), thickness=5))
        #     faces_img.append(cv2.rectangle(img2, (x, y), (x + w, y + h), (255, 0, 0), thickness=5))
            faces_img.append(cv2.resize(img2[y: y + h, x: x + w], dsize=(224, 224)))
        changed_imgs.append(new_face)

    return faces_img, changed_imgs


def get_img(img_name, dir_path='input_photo/', add_noize=False, db_model=None):
    faces = preprocess_photo(img_name, dir_path=dir_path, add_noize=add_noize, db_model=db_model)
    print("Лиц обнаружено: {}".format(len(faces[0])))

    return faces
