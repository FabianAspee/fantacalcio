import time
from abc import ABC

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from common.GeneralFunction import GeneralFunction

from project.project.Graphic import Graphic


class RandomForestRegressorModel(GeneralFunction):

    def create_variable_for_model(self, player, windows_size=13):
        return super().create_variable_for_model(player, windows_size)

    def __init__(self):
        self.__graphic__ = Graphic()

    def start(self, player, name_player):
        new_x, target_x, new_y, target_y = self.create_variable_for_model(player)
        model = self.__create_model__()
        self.__train__(model, new_x, target_x, new_y, target_y)
        y_predict = self.__test__(model, new_x[new_x.shape[0] - 1, :].reshape(1, -1),
                                  target_y, len(new_y))
        self.__graphic__.plot_mlp(target_x, target_y, y_predict, name_player)

    @staticmethod
    def __create_model__():
        random_forest = RandomForestRegressor(n_estimators=100, criterion='mse', max_depth=None,
                                              min_samples_split=2, min_samples_leaf=1,
                                              min_weight_fraction_leaf=0.01, max_features='auto',
                                              max_leaf_nodes=None, min_impurity_decrease=0.0,
                                              min_impurity_split=None, bootstrap=True, oob_score=False,
                                              n_jobs=None, random_state=None, verbose=2, warm_start=False)
        return random_forest

    @staticmethod
    def __train__(random_forest: RandomForestRegressor, new_x, target_x, new_y, target_y):
        print('Training ...')

        time_start = time.time()
        kf = KFold(n_splits=10)
        for train_indices, test_indices in kf.split(new_x):
            X_train, X_test = new_x[train_indices], new_x[test_indices]
            Y_train, Y_test = target_x[train_indices], target_x[test_indices]
            random_forest.fit(X_train, Y_train.ravel())
            y_pred = random_forest.predict(X_test)
            print('Predic', y_pred)
            print('MSE =', mean_squared_error(Y_test, y_pred))
            print('R2 =', r2_score(Y_test, y_pred))
            print(random_forest.score(X_train, Y_train))

    def __test__(self, random_forest: RandomForestRegressor, x_test, y_test, period):
        y_predict_cumulative = np.empty((0, 1))
        for val in range(period):
            y_predict = random_forest.predict(x_test)
            x_test = self.__create_new_window__(x_test, y_predict)
            y_predict_cumulative = np.append(y_predict_cumulative, y_predict)
        print(y_predict_cumulative)
        print('MSE =', mean_squared_error(y_test, y_predict_cumulative))
        print(r2_score(y_test, y_predict_cumulative))
        return y_predict_cumulative
