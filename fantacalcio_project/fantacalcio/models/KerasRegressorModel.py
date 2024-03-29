from abc import ABC

import numpy as np
from keras import backend as K
from keras.layers import Dense, Dropout, Activation
from keras.models import Sequential
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor

from common.GeneralFunction import GeneralFunction
from project.project.Graphic import Graphic

from common.CommonFunction import CommonFunction


class KerasRegressorModel(GeneralFunction):

    def create_variable_for_model(self, player, windows_size=13):
        return super().create_variable_for_model(player, windows_size)

    def __init__(self):
        self.__graphic__ = Graphic()

    def start(self, player, name_player):
        new_x, target_x, new_y, target_y = self.create_variable_for_model(player)
        model = self.__create_model__()
        self.__train__(model, new_x, target_x, new_y, target_y)
        y_predict = self.__test__(model, new_x[new_x.shape[0] - 1, :].reshape(1, -1),
                                  target_y, len(new_y), new_x, new_y)
        self.__graphic__.plot_mlp(target_x, target_y, y_predict, name_player)

    @staticmethod
    def __create_model__():
        return KerasRegressor(build_fn=CommonFunction.__baseline_model__, epochs=200, batch_size=5, verbose=2)

    @staticmethod
    def __train__(keras: KerasRegressor, new_x, target_x, new_y, target_y):
        kf = KFold(n_splits=10)
        for train_indices, test_indices in kf.split(new_x):
            x_train, x_test = new_x[train_indices], new_x[test_indices]
            y_train, y_test = target_x[train_indices], target_x[test_indices]
            keras.fit(x_train, y_train.ravel())
            y_pred = keras.predict(x_test)
            print('Predic', y_pred)
            print('MSE =', mean_squared_error(y_test, y_pred))
            print('R2 =', r2_score(y_test, y_pred))

    def __test__(self, keras: KerasRegressor, x_test, y_test, period, new_x, new_y):
        y_predict_cumulative = np.empty((0, 1))
        for val in range(period):
            y_predict = keras.predict(x_test)
            x_test = self.__create_new_window__(x_test, y_predict)
            y_predict_cumulative = np.append(y_predict_cumulative, y_predict)
        print(y_predict_cumulative)
        print('MSE =', mean_squared_error(y_test, y_predict_cumulative))
        print(r2_score(y_test, y_predict_cumulative))
        return y_predict_cumulative
