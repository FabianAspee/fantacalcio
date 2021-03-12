import time
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from fantacalcio.common.GeneralFunction import GeneralFunction
from fantacalcio.project.project.Graphic import Graphic


class MLPRegressorModel(GeneralFunction):

    def __create_new_window__(self, x_test, y_predict):
        return super().__create_new_window__(x_test, y_predict)

    def compute_windows(self, x_train_aux, n_past=20):
        return super().compute_windows(x_train_aux, n_past)

    def __init__(self):
        self.__graphic__ = Graphic()

    def __create_variable_for_model(self, player, windows_size=13):
        windows, player_vote = self.compute_windows(player, windows_size)
        x, y = windows[:int(len(windows) * 0.8)], windows[int(len(windows) * 0.8):]
        return x, player_vote[:int(len(player_vote) * 0.8)], y, player_vote[
                                                                int(len(player_vote) * 0.8):]

    def start(self, player, name_player):
        new_x, target_x, new_y, target_y = self.__create_variable_for_model(player)
        model = self.__create_model__(len(target_x))
        self.__train__(model, new_x, target_x, new_y, target_y)
        y_predict = self.__test__(model, new_x[new_x.shape[0] - 1, :].reshape(1, -1),
                                  target_y, len(new_y))
        self.__graphic__.plot_mlp(target_x, target_y, y_predict, name_player)

    @staticmethod
    def __create_model__(len_train):
        mlp = MLPRegressor(hidden_layer_sizes=(20,),
                           activation='relu',
                           solver='adam',
                           learning_rate_init=0.0009,
                           warm_start=True,
                           max_iter=500,
                           batch_size=int(len_train * 0.2),
                           alpha=0.0009,
                           random_state=1234)
        return mlp

    @staticmethod
    def __train__(mlp: MLPRegressor, new_x, target_x, new_y, target_y):
        max_epochs = 60
        print('Training ...')
        time_start = time.time()
        epochs_training_loss = []
        epochs_validation_accuracy = []
        epochs_training_accuracy = []
        for i in range(max_epochs):
            mlp.fit(new_x, target_x)
            epochs_training_loss.append(mlp.loss_)
            epochs_training_accuracy.append(mlp.score(new_x, target_x) * 100)
            epochs_validation_accuracy.append(mlp.score(new_y, target_y) * 100)
            print("Epoch %2d: Loss = %5.4f, TrainAccuracy = %4.2f%%, ValidAccuracy = %4.2f%%" % (
                i + 1, epochs_training_loss[-1], epochs_training_accuracy[-1], epochs_validation_accuracy[-1]))

        print('Total Time: %.2f sec' % (time.time() - time_start))

        max_valacc_idx = np.array(epochs_validation_accuracy).argmax()
        print('Max Accuracy on Validation = %4.2f%% at Epoch = %2d' % (
            epochs_validation_accuracy[max_valacc_idx], max_valacc_idx + 1))

    def __test__(self, mlp: MLPRegressor, x_test, y_test, period):
        y_predict_cumulative = np.empty((0, 1))
        for val in range(period):
            y_predict = mlp.predict(x_test)
            x_test = self.__create_new_window__(x_test, y_predict)
            y_predict_cumulative = np.append(y_predict_cumulative, y_predict)
        print(y_predict_cumulative)
        print('MSE =', mean_squared_error(y_test, y_predict_cumulative))
        print(r2_score(y_test, y_predict_cumulative))
        return y_predict_cumulative
