import matplotlib.pyplot as plt
import numpy as np
from fantacalcio.common.CommonFunction import CommonFunction

plt.rcParams["figure.figsize"] = (18, 6)


class Graphic:
    def __init__(self):
        super().__init__()

    @staticmethod
    def plot_figure(data, date, name_player, name_team=None):
        data[data == '-'] = np.NaN
        plt.plot(data.astype(np.float))
        plt.savefig(CommonFunction.total_path(CommonFunction.base_path, f"../resources/{name_player}-{name_team}"),
                    format='svg')

    @staticmethod
    def plot_figure_by_year(data, date, name_player, name_team=None):
        init = 0
        finish = 38
        fig, ax = plt.subplots()
        for year in np.unique(date):
            ax.plot([v for v in range(init, finish)], data[init:finish].astype(np.float), label=year)
            init = finish
            finish = finish + finish if (finish + finish) < len(data) else len(data)
        legend = ax.legend(loc='upper center', shadow=False, fontsize='x-large')
        legend.get_frame().set_facecolor('C0')
        plt.savefig(CommonFunction.total_path(CommonFunction.base_path, f"../resources/{name_player}-{name_team}"),
                    format='svg')

    @staticmethod
    def plot_figure_ltms(train, test, n_input, train_pred, predictions, name_player):
        plt.plot(np.concatenate((train, test)), label='votes')
        plt.plot([None for x in range(n_input)] + [x for x in train_pred],
                 color='brown', label="train_prediction")
        plt.plot([None for t in range(len(train))] + [x for x in predictions],
                 label="prediction", color='darkgreen')
        plt.savefig(CommonFunction.total_path(CommonFunction.base_path, f"../resources/result_prediction_ltms_"
                                                                        f"{name_player}"))

    @staticmethod
    def plot_arima(train, test, train_pred, forecasts, confint, name_player):
        plt.plot(np.concatenate((train, test)))
        plt.plot([index for index in range(len(train_pred))], train_pred, color='brown')
        plt.plot([index for index in range(len(train_pred), len(train_pred) + len(forecasts))], forecasts,
                 color='darkgreen')
        plt.fill_between([index for index in range(len(confint[:, 1]))],
                         confint[:, 0],
                         confint[:, 1],
                         color='k', alpha=.15)

        plt.title("ARIMA - Final Forecast " + name_player)
        plt.savefig(CommonFunction.total_path(CommonFunction.base_path, f"../resources/result_prediction_arima_"
                                                                        f"{name_player}"))

    @staticmethod
    def plot_sarima(train, test, train_pred, forecasts, name_player):
        plt.plot(np.concatenate((train, test)))
        plt.plot([index for index in range(len(train_pred))], train_pred, color='brown')
        plt.plot([index for index in range(len(train_pred), len(train_pred) + len(forecasts))], forecasts,
                 color='darkgreen')

        plt.title("SARIMA - Final Forecast " + name_player)
        plt.savefig(CommonFunction.total_path(CommonFunction.base_path, f"../resources/result_prediction_sarima_"
                                                                        f"{name_player}"))

    @staticmethod
    def plot_mlp(x_train, y_test, y_pred, name_player):
        fig, ax = plt.subplots(figsize=(16, 6))
        plt.title('MLPRegression Real VS Predict')
        ax.plot([x for x in x_train] + [y for y in y_test], '-b', label='Real')
        ax.plot([None for _ in x_train] + [y for y in y_pred], '-r', label='Predict')
        leg = ax.legend()
        fig.savefig(CommonFunction.total_path(CommonFunction.base_path, f"../resources/result_prediction_mlp_"
                                                                        f"{name_player}"))
