import cv2
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import os
from shutil import copyfile
import time


import data
import metrics
import model
import preprocessing


def verify_face(img, db, vgg_model, k, mode='predict'):
    img_representation = vgg_model.predict(img)[0, :]
    if mode == 'predict':
        return img_representation, [], [], []
    elif mode == 'cmp':
        cosine_similarity = np.array([metrics.find_cosine_similarity(img_representation, db_img_representation) for db_img_representation in db.data])
        euclidean_distance = np.array([metrics.find_euclidean_distance(img_representation, db_img_representation) for db_img_representation in db.data])
        knn2img = np.argpartition(euclidean_distance, range(k))[:k]
        return img_representation, knn2img, cosine_similarity, euclidean_distance
    else:
        raise Exception('Unknown mode')


def show_similar(face, photo_ids, db_photo_dir, metric, k):
    fig, imgs = plt.subplots(1, k, num='{k} nearest photos'.format(k=k))
    for ax in imgs:
        ax.set_xticks([])
        ax.set_yticks([])
    imgs[0].imshow(face)
    imgs[0].set_title('face2find')
    for i in range(1, min(k, len(photo_ids))):
        imgs[i].imshow(image.load_img(db_photo_dir+'{}.jpg'.format(photo_ids[i - 1])))
        imgs[i].set_title(metric[i - 1])
    plt.show()


input_photo_dir = 'input_photo/'
database_photo_dir = 'db_photo/'


def main():
    mode = int(input('Enter:\n\t1 - to add photos to DB\n\t2 - to add and show nearest photos\n\t3 - only to show nearest photos\n'))
    if mode not in (1, 2, 3):
        raise Exception('Wrong execution mode')

    k_nearest = 0
    if mode == 2 or mode == 3:
        k_nearest = int(input('Enter K:\n'))

    db = data.DataBase()
    vgg_model = model.get_vgg_model()
    cur_time = time.time() - 5

    while True:
        if time.time() - cur_time < 5:
            continue
        cur_time = time.time()

        temp_list_dir = os.listdir(input_photo_dir)
        for file in temp_list_dir:
            if file[0] == '.':
                continue
            print('Working with {}'.format(file))
            face_imgs, faces = preprocessing.get_img(file, input_photo_dir)

            for i in range(len(faces)):
                img_descriptor, knn2img, cosine_metric, euclidean_metric = verify_face(faces[i], db, vgg_model,
                                                                                       k=k_nearest,
                                                                                       mode='predict' if mode == 1 else 'cmp')
                if mode == 2 or mode == 3:
                    show_similar(face_imgs[i], knn2img, database_photo_dir, euclidean_metric[knn2img], k_nearest)
                if mode == 3:
                    continue
                db.append(img_descriptor)
                copyfile(input_photo_dir + file, database_photo_dir + '{}.jpg'.format(db.size - 1))
            os.remove(input_photo_dir + file)
        if cv2.waitKey() == ord('q'):
            return


if __name__ == '__main__':
    main()


