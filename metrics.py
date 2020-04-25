import numpy as np



def find_cosine_similarity(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))


def find_euclidean_distance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance


def avgPrK(img_representation, name, db, k):
    euclidean_distance = np.array([find_euclidean_distance(img_representation, db_img_representation) for db_img_representation in db.data])
    kn = np.argpartition(euclidean_distance, range(min(k, db.size)))[:k]
    same = 0
    i = 0
    res = 0
    for idx in kn:
        if db.names[idx] == name:
            same += 1
            res += same / i
    res *= 1 / k
    return res


def MAP(db, k):
    res = 0
    n = len(db)
    for i in range(n):
        res += avgPrK(db.data[i], db.names[i], db, k)
    res *= 1 / n
    return res
