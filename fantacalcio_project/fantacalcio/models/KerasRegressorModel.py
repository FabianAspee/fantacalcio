import numpy as np
from keras import backend as K
from keras.layers import Dense, Dropout, Activation
from keras.models import Sequential
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor

from common.GeneralFunction import GeneralFunction
from project.project.Graphic import Graphic


class KerasRegressorModel(GeneralFunction):

    def __create_new_window__(self, x_test, y_predict):
        return super().__create_new_window__(x_test, y_predict)

    def compute_windows(self, x_train_aux, n_past=20):
        return super().compute_windows(x_train_aux, n_past)

    def __init__(self):
        self.__graphic__ = Graphic()

    def __create_variable_for_model(self, player, windows_size=10):
        windows, player_vote = self.compute_windows(player, windows_size)
        x, y = windows[:int(len(windows) * 0.8)], windows[int(len(windows) * 0.8):]
        return x, player_vote[:int(len(player_vote) * 0.8)], y, player_vote[int(len(player_vote) * 0.8):]

    def start(self, player, name_player):
        new_x, target_x, new_y, target_y = self.__create_variable_for_model(player)
        model = self.__create_model__()
        self.__train__(model, new_x, target_x, new_y, target_y)
        y_predict = self.__test__(model, new_x[new_x.shape[0] - 1, :].reshape(1, -1),
                                  target_y, len(new_y), new_x, new_y)
        self.__graphic__.plot_mlp(target_x, target_y, y_predict, name_player)

    @staticmethod
    def __coefficient_determination__(y_true, y_predict):
        ss_res = K.sum(K.square(y_true - y_predict))
        ss_tot = K.sum(K.square(y_true - K.mean(y_true)))
        return 1 - ss_res / (ss_tot + K.epsilon())

    def __baseline_model__(self):
        model = Sequential()
        n_input = 10
        n_hidden = 14
        n_output = 1
        model.add(Dense(100, input_shape=(n_input,), activation='relu'))  # hidden neurons, 1 layer
        model.add(Dropout(0.5))
        model.add(Dense(
            units=50))
        model.add(Dense(n_output))
        model.add(Activation("linear"))
        model.compile(loss='mean_squared_error', optimizer='adam',
                      metrics=[self.__coefficient_determination__])
        model.summary()
        return model

    def __create_model__(self):
        return KerasRegressor(build_fn=self.__baseline_model__, epochs=200, batch_size=5, verbose=2)

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
