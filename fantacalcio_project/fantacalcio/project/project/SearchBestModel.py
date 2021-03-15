import asyncio

import statsmodels.api as sm
from common.CommonFunction import CommonFunction
from common.GeneralFunction import GeneralFunction
from common.ThreadPool import ThreadPool
from models.LSTMModel import LSTMModel
from models.commonmodel.CommonGridSearch import CommonGridSearch
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from statsmodels.sandbox.tsa import movmean
from statsmodels.sandbox.tsa.try_arma_more import pm
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor
import json
import os
import math


class SearchBestModel(GeneralFunction):

    def create_variable_for_model(self, player, windows_size=13):
        return super().create_variable_for_model(player, windows_size)

    @staticmethod
    def __define_processor__(number_grid):
        number_processor = os.cpu_count()
        result_div_for_process = int(math.floor(number_processor / number_grid))
        result_mod_for_process = int(number_processor % number_grid)
        list_processor = [result_div_for_process for _ in range(0, number_grid)]
        for index in range(0, result_mod_for_process):
            list_processor[index] = list_processor[index] + 1
        return list_processor

    async def start(self, info_player, name_player):
        use_all_processor = -1
        tasks = [self.lstm_regressor_grid(info_player, name_player, use_all_processor)]
        await asyncio.wait(tasks)

        return ['Keras', 'DecisionTree', 'RandomForest']

    async def mlp_regressor_grid(self, info_player, name_player, num_processor):
        mlp_model = MLPRegressor()
        parameters = {'hidden_layer_sizes': [(20,), (40,)],
                      'activation': ("relu", "identity"),
                      'solver': ("adam", "adam"),
                      'learning_rate_init': [0.0009],
                      'warm_start': [True],
                      'max_iter': [500, 1000],
                      'batch_size': [5, 10, 15],
                      'alpha': [0.0009],
                      'random_state': [1234]}
        x_train, x_target, _, _ = self.__get_information_for_model__(info_player)
        get_best_params = CommonGridSearch.get_best_params(mlp_model, parameters, x_train, x_target, num_processor)
        self.__save_best_model_(get_best_params, 'mlp')

    async def decision_tree_regressor_grid(self, info_player, name_player, num_processor):
        print('decision')
        decision_tree_model = DecisionTreeRegressor()
        parameters = {'criterion': ('mse', 'mae'),
                      'splitter': ('best', 'random'),
                      'min_samples_split': [2, 4, 5, 7, 8],
                      'min_samples_leaf': [2, 4, 5, 7, 8, 100, 120, 200],
                      'min_weight_fraction_leaf': [0.002, 0.005],
                      'max_features': ('auto', 'log2'),
                      'random_state': [True, False],
                      'min_impurity_decrease': [0.02, 0.05, 0.002], 
                      'ccp_alpha': [0.02, 0.05, 0.002]}
        x_train, x_target, _, _ = self.__get_information_for_model__(info_player)
        get_best_params = CommonGridSearch.get_best_params(decision_tree_model, parameters, x_train, x_target,
                                                           num_processor)
        self.__save_best_model_(get_best_params, 'decision_tree')

    async def random_forest_regressor_grid(self, info_player, name_player, num_processor):
        print('random')
        random_forest_model = RandomForestRegressor()
        parameters = {'n_estimators': [100, 200, 50, 250],
                      'criterion': ('mse', 'mae'),
                      'min_samples_split': [2, 4, 6, 7, 5],
                      'min_samples_leaf': [1, 3, 5, 7],
                      'min_weight_fraction_leaf': [0.01, 0.05, 0.001],
                      'max_features': ('auto', 'log2'),
                      'min_impurity_decrease': [0.02, 0.05, 0.002],
                      'bootstrap': [True, False],
                      'oob_score': [True, False],
                      'warm_start': [True, False],
                      'ccp_alpha': [0.02, 0.05, 0.002]}
        x_train, x_target, _, _ = self.__get_information_for_model__(info_player)
        get_best_params = CommonGridSearch.get_best_params(random_forest_model, parameters, x_train, x_target,
                                                           num_processor)
        self.__save_best_model_(get_best_params, 'random_forest')

    async def lstm_regressor_grid(self, info_player, name_player, num_processor):
        lstm_model = LSTMModel()
        parameters = {}
        x_train, x_target, _, _ = self.__get_information_for_model__(info_player)
        return CommonGridSearch.get_all_result(lstm_model, parameters, x_train, x_target, num_processor)

    async def sarima_regressor_grid(self, info_player, name_player, num_processor):
        x_train, x_target, _, _ = self.__get_information_for_model__(info_player)
        sarima_model = sm.tsa.SARIMAX(info_player)
        parameters = {'order': [(1, 1, 1), (1, 2, 3), (2, 2, 1)],
                      'seasonal_order': [(0, 1, 1, 4), (1, 1, 1, 2), (3, 1, 2, 4)],
                      'tren': ['n', 'c', 't', 'ct'],
                      'enforce_stationarity': [True, False],
                      'enforce_invertibility': [True, False],
                      'hamilton_representation': [True, False],
                      'concentrate_scale': [True, False]}
        get_best_params = CommonGridSearch.get_best_params(sarima_model, parameters, x_train, x_target, num_processor)
        self.__save_best_model_(get_best_params, 'sarima')

    async def arima_regressor_grid(self, info_player, name_player, num_processor):
        arima_model = pm.auto_arima()
        parameters = {}
        x_train, x_target, _, _ = self.__get_information_for_model__(info_player)
        return CommonGridSearch.get_all_result(arima_model, parameters, x_train, x_target, num_processor)

    async def keras_regressor_grid(self, info_player, name_player, num_processor):
        print('keras')
        keras_regressor_model = KerasRegressor(CommonFunction.__baseline_model__)
        parameters = {
            'dense': [100, 50, 200],
            'activation_init': ('relu', 'sigmoid ', 'softmax ', 'tanh'),
            'units': [50, 40, 100],
            'activation': ('relu', 'sigmoid ', 'softmax ', 'tanh'),
            'optimizer': ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'],
            'epochs': [10, 100, 200],
            'batch_size': [5, 10, 15],
            'rate': [0.5, 0.4, 0.3, 0.2, 0.1, 0]
        }
        x_train, x_target, _, _ = self.__get_information_for_model__(info_player)
        get_best_params = CommonGridSearch.get_best_params(keras_regressor_model, parameters, x_train, x_target,
                                                           num_processor)
        self.__save_best_model_(get_best_params, 'keras_regressor')

    async def svr_regressor_grid(self, info_player, name_player, num_processor):
        svr_regressor_model = SVR()
        parameters = {
            'kernel': ('linear', 'poly', 'rbf', 'sigmoid'),
            'degree': [3, 5, 7, 9],
            'gamma': ('scale', 'auto'),
            'coef0': [0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
            'tol': [1e-3, 1e-4, 1e-5, 1e-6, 1e-7],
            'C': [1, 2, 1.5, 1.9, 0.5, 0.05],
            'epsilon': [0.5, 0.005, 0.02, 0.002],
            'shrinking': [True, False],
            'cache_size': [200, 400, 600]
        }
        x_train, x_target, _, _ = self.__get_information_for_model__(info_player)
        get_best_params = CommonGridSearch.get_best_params(svr_regressor_model, parameters, x_train, x_target,
                                                           num_processor)
        self.__save_best_model_(get_best_params, 'svr_regressor')

    def __get_information_for_model__(self, info_player):
        return self.create_variable_for_model(SearchBestModel.__standardize__(SearchBestModel.__mov_mean__(
            info_player)))

    @staticmethod
    def __mov_mean__(player):
        return movmean(player, 6, lag='centered')

    @staticmethod
    def __standardize__(x_train):
        result, _ = CommonFunction.standardize(x_train)
        return result

    @staticmethod
    def __save_best_model_(model, name_model):
        json_structure = json.dumps(model)
        file = open(CommonFunction.total_path(CommonFunction.base_path, f'../resources/model-{name_model}.json'), "w")
        file.write(json_structure)
        file.close()
