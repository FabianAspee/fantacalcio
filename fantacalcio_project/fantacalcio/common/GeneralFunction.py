from abc import ABC, abstractmethod
import numpy as np


class GeneralFunction(ABC):
    @abstractmethod
    def compute_windows(self, np_array, n_past=1):
        data_x, data_y = [], []
        for i in range(len(np_array) - n_past - 1):
            a = np_array[i:(i + n_past), 0]
            data_x.append(a)
            data_y.append(np_array[i + n_past, 0])
        return np.array(data_x), np.array(data_y)

    @abstractmethod
    def __create_new_window__(self, x_test, y_predict):
        new_x_test = np.append(x_test[:, 1:], y_predict).reshape(1, -1)
        return new_x_test