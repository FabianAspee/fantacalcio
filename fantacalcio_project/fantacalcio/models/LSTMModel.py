from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy as np
from keras.preprocessing.sequence import TimeseriesGenerator
from project.project.Graphic import Graphic

np.random.seed(458)


class LSTMModel:

    def __init__(self, n_input=10):
        self.__lstm_model__ = Sequential()
        self.__n_inputs = n_input
        self.__accuracies__ = {}
        self.__players_models_lstm__ = {}
        self.create_model()
        self.__graphic__ = Graphic()

    def start(self, player, name_player):
        x, y = player[:int(len(player) * 0.9)], player[int(len(player) * 0.9):]
        self.__fit__(x)
        prediction_y = self.__predictive__(y, name_player)
        prediction_x = self.__predictive__(x, name_player)
        self.__mean__()
        self.__graphic__.plot_figure_ltms(x, y, self.__n_inputs, prediction_x, prediction_y, name_player)

    def create_model(self):
        self.__lstm_model__.add(LSTM(20, activation='relu', input_shape=(self.__n_inputs, 1), dropout=0.05))
        self.__lstm_model__.add(Dense(1))
        self.__lstm_model__.compile(optimizer='adam', loss='mse')

    def __fit__(self, train):
        print(f"Training ")
        train_generator = TimeseriesGenerator(train, train, length=self.__n_inputs, batch_size=1)
        # lstm_model.summary()
        self.__lstm_model__.fit(train_generator, epochs=50)

    def __predictive__(self, test, name_player):
        test_generator = TimeseriesGenerator(test, test, length=self.__n_inputs, batch_size=1)
        predictions = self.__lstm_model__.predict(test_generator).flatten()
        self.__accuracies__[name_player] = self.__forecast_accuracy__(test[self.__n_inputs:].flatten(), predictions)
        self.__players_models_lstm__[name_player] = {
            'model': self.__lstm_model__,
            'rmse': self.__accuracies__[name_player]['rmse']
        }
        return predictions

    def __forecast_accuracy__(self, forecast, actual):
        mape = np.mean(np.abs(forecast - actual) / np.abs(actual))  # MAPE
        me = np.mean(forecast - actual)  # ME
        mae = np.mean(np.abs(forecast - actual))  # MAE
        mpe = np.mean((forecast - actual) / actual)  # MPE
        rmse = np.mean((forecast - actual) ** 2) ** .5  # RMSE
        # corr = np.corrcoef(forecast, actual)[0,1]   # corr
        mins = np.amin(np.hstack([forecast[:, None],
                                  actual[:, None]]), axis=1)
        maxs = np.amax(np.hstack([forecast[:, None],
                                  actual[:, None]]), axis=1)
        minmax = 1 - np.mean(mins / maxs)  # minmax
        return ({'mape': mape, 'me': me, 'mae': mae,
                 'mpe': mpe, 'rmse': rmse,
                 'corr': 0, 'minmax': minmax})

    def __mean__(self):
        lstm_mean_rsme = np.mean([self.__players_models_lstm__[x]['rmse'] for x in self.__players_models_lstm__])
        print("LSTM mean RSME {}".format(lstm_mean_rsme))


