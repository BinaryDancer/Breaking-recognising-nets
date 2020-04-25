import cv2
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from foolbox.attacks import FGSM
import numpy as np
import matplotlib.pyplot as plt
import metrics
import tensorflow as tf
import foolbox
tf.config.list_physical_devices('GPU')
# tf.compat.v1.disable_eager_execution()
# tf.compat.v1.experimental.output_all_intermediates(True)

from foolbox.criteria import Misclassification


face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')


def preprocess_photo(img_name, dir_path='input_photo/', add_noize=False, db_model=None):
    image_path = dir_path + img_name
    img = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        img_gray,
        scaleFactor=1.1,
        minNeighbors=25,
        minSize=(5, 5)
    )

    faces_img = []
    changed_imgs = []
    for (x, y, w, h) in faces:
        img2 = img.copy()
        if add_noize:
            foolbox_model = foolbox.TensorFlowModel(model=db_model[1], bounds=(0, 255))
            tmp_face = np.expand_dims(cv2.resize(img2[y: y + h, x: x + w], dsize=(224, 224)), axis=0)
            attack = foolbox.attacks.FGSM()
            # attack = foolbox.attacks.L2DeepFoolAttack()

            print('real class: {}'.format([np.argmax(db_model[1].predict(tmp_face)),]))
            new_face = attack(foolbox_model, np.array([tmp_face]), Misclassification(np.array([db_model[1].predict(tmp_face).argmax()])), epsilons=0.0007)
            # new_face = attack(foolbox_model, np.array([tmp_face,]), np.array([np.argmax(db_model[1].predict(tmp_face)),]), epsilons=0.0007)
            # new_face = attack(foolbox_model, tmp_face,)
            # new_face = attack(tmp_face, labels=np.ndarray(1))
            print('new class: {}'.format(db_model[1].predict(new_face)))
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
