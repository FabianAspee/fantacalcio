import numpy as np
import statsmodels.api as sm

from fantacalcio.project.project.Graphic import Graphic


class SarimaModel:
    __DEFAULT_PEARSON_VALUE__ = 0

    def __init__(self):
        self.__graphic__ = Graphic()

    def start(self, player, name_player):
        x, y = player[:int(len(player) * 0.6)], player[int(len(player) * 0.6):]
        model = self.__construct_model__(player, x)
        result, forecast = self.__predictive__(model, x, y)
        self.__graphic__.plot_sarima(x, y, result['x_predict'], forecast, name_player)

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
        s_model = sm.tsa.SARIMAX(train, order=(1, 1, 1),
                                 seasonal_order=(0, 1, 1, 4))

        return s_model.fit()

    def __predictive__(self, s_model, train, test):
        x_predict = s_model.get_prediction(start=0, end=len(train)).predicted_mean
        forecasts = s_model.forecast(steps=len(test), return_conf_int=True)
        metrics = self.__forecast_accuracy__(forecasts, test.reshape(-1, ))
        return {'model': s_model, 'rmse': metrics['rmse'], 'x_predict': x_predict}, forecasts

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
