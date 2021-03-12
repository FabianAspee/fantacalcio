import pmdarima as pm
import numpy as np
from project.project.Graphic import Graphic


class ArimaModel:
    __DEFAULT_PEARSON_VALUE__ = 0

    def __init__(self):
        self.__graphic__ = Graphic()

    def start(self, player, name_player):
        x, y = player[:int(len(player) * 0.6)], player[int(len(player) * 0.6):]
        model = self.__construct_model__(player, x)
        result, forecast, confint = self.__predictive__(model, x, y)
        self.__graphic__.plot_arima(x, y, result['x_predict'], forecast, confint, name_player)

    def __pearson__(self, player):
        pearson = np.corrcoef(player)
        if len(pearson) > 1:
            pearson = pearson[1:]  # exclude the first element (always 1)
            max_id = pearson.argmax()
            return max_id + 1, pearson[max_id]
        else:
            return self.__DEFAULT_PEARSON_VALUE__, self.__DEFAULT_PEARSON_VALUE__

    def __construct_model__(self, player_sequence, train):
        _m = self.__pearson__(player_sequence)[0]
        print(f"Testing SARIMA with max seasonality {_m}")
        # _the following VERY heuristic patch
        if _m <= 4:
            _m = 12
        # Seasonal - fit stepwise auto-ARIMA, returns an ARIMA model
        s_model = pm.auto_arima(train, start_p=1, start_q=1,
                                test='adf',
                                max_p=3, max_q=3, m=_m,
                                start_P=0, seasonal=True,
                                d=0, D=1, trace=True,
                                error_action='ignore',
                                suppress_warnings=True,
                                stepwise=True)
        return s_model

    def __predictive__(self, s_model, train, test):
        x_predict = s_model.predict_in_sample(start=0, end=len(train))
        forecasts, confint = s_model.predict(n_periods=len(test), return_conf_int=True)
        metrics = self.__forecast_accuracy__(forecasts, test.reshape(-1, ))
        return {'model': s_model, 'rmse': metrics['rmse'], 'x_predict': x_predict}, forecasts, confint

    @staticmethod
    def __forecast_accuracy__(forecast, actual):
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
