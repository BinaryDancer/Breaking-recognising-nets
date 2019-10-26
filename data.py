import numpy as np


class DataBase:
    def __init__(self, path='database.txt'):
        self._size = 0
        self._data = []
        self._path = path
        with open(path, 'r') as f:
            for line in f:
                self._data.append(np.array(list(map(float, line.split()))))
                self._size += 1

    @property
    def data(self):
        return self._data

    @property
    def size(self):
        return self._size

    def append(self, data: np.ndarray):
        self._data.append(data)
        self._size += 1
        with open(self._path, 'a') as f:
            f.write(' '.join(map(str, data)) + '\n')
