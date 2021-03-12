from statsmodels.sandbox.tsa.movstat import movmean

from fantacalcio.models.ArimaModel import ArimaModel
from fantacalcio.common.CommonFunction import CommonFunction
from fantacalcio.models.KerasRegressorModel import KerasRegressorModel
from fantacalcio.models.LSTMModel import LSTMModel
from fantacalcio.models.MLPRegressorModel import MLPRegressorModel
from fantacalcio.common.ThreadPool import ThreadPool
from fantacalcio.models.SarimaModel import SarimaModel


class PredictiveOperation:

    def __init__(self, model='MLP', num_iteration=2):
        self.__model__ = model
        self.__num_iteration__ = num_iteration

    def start(self, player, name_player, model=None):
        self.__model__ = model if model is not None else self.__model__
        if self.__model__ == 'Sarima':
            self.__start_regressor__(SarimaModel, self.__standardize__(self.mov_mean(player)), name_player)

        elif self.__model__ == 'Arima':
            self.__start_regressor__(ArimaModel, self.__standardize__(self.mov_mean(player)), name_player)

        elif self.__model__ == 'LSTM':
            self.__start_regressor__(LSTMModel, self.__standardize__(self.mov_mean(player)), name_player)

        elif self.__model__ == 'MLP':
            self.__start_regressor__(MLPRegressorModel, self.__standardize__(self.mov_mean(player)), name_player)

        elif self.__model__ == 'Keras':
            self.__start_regressor__(KerasRegressorModel, self.__standardize__(self.mov_mean(player)), name_player)

    @staticmethod
    def __start_regressor__(class_execute, player, name_player):
        class_execute().start(player, name_player)

    @staticmethod
    def mov_mean(player):
        return movmean(player, 6, lag='centered')

    @staticmethod
    def __standardize__(x_train):
        result, _ = CommonFunction.standardize(x_train)
        return result

    @staticmethod
    def __normalize__(x_train):
        result, _ = CommonFunction.normalize(x_train)
        return result

    @staticmethod
    def __min_max_Scaler__(x_train):
        result, _ = CommonFunction.min_max_Scaler(x_train)
        return result
