import os
import numpy as np
from sklearn import preprocessing
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Activation
from keras import backend as k


class CommonFunction:
    base_path = os.path.dirname(__file__)
    total_path = os.path.join

    @staticmethod
    def compute_windows(np_array, n_past=1):
        data_x, data_y = [], []
        for i in range(len(np_array) - n_past - 1):
            a = np_array[i:(i + n_past), 0]
            data_x.append(a)
            data_y.append(np_array[i + n_past, 0])
        return np.array(data_x), np.array(data_y)

    @staticmethod
    def __create_new_window__(self, x_test, y_predict):
        new_x_test = np.append(x_test[:, 1:], y_predict).reshape(1, -1)
        return new_x_test

    @staticmethod
    def standardize(x_train):
        standard = preprocessing.StandardScaler()
        return standard.fit_transform(x_train), standard

    @staticmethod
    def normalize(x_train):
        normalizer = preprocessing.Normalizer()
        return normalizer.fit_transform(x_train), normalizer

    @staticmethod
    def min_max_Scaler(x_train):
        min_max = preprocessing.MinMaxScaler()
        return min_max.fit_transform(x_train), min_max

    @staticmethod
    def __coefficient_determination__(y_true, y_predict):
        ss_res = k.sum(k.square(y_true - y_predict))
        ss_tot = k.sum(k.square(y_true - k.mean(y_true)))
        return 1 - ss_res / (ss_tot + k.epsilon())

    @staticmethod
    def __baseline_model__(dense=100, activation_init='relu', rate=0.5, units=50, activation='relu', optimizer='adam'):
        model = Sequential()
        n_input = 13
        n_hidden = 14
        n_output = 1
        model.add(Dense(dense, input_shape=(n_input,), activation=activation_init))  # hidden neurons, 1 layer
        model.add(Dropout(rate))
        model.add(Dense(
            units=units))
        model.add(Dense(n_output))
        model.add(Activation(activation))
        model.compile(loss='mean_squared_error', optimizer=optimizer,
                      metrics=[CommonFunction.__coefficient_determination__])
        model.summary()
        return model
