import numpy as np


class DataBase:
    def __init__(self, path='database.txt'):
        self._size = 0
        self._data = []
        self._names = []
        self._path = path
        with open(path, 'r') as f:
            for line in f:
                self._names.append(line.split()[-1])
                self._data.append(np.array(list(map(float, line.split()[:-1]))))
                self._size += 1

    @property
    def data(self):
        return self._data

    @property
    def names(self):
        return self._names

    @property
    def size(self):
        return self._size

    def append(self, data: np.ndarray, name):
        self._data.append(data)
        self._names.append(name)
        self._size += 1
        with open(self._path, 'a') as f:
            f.write(' '.join(map(str, data)) + ' ' + name + '\n')
