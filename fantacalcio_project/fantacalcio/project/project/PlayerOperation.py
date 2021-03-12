import numpy as np
import pandas as pd

from fantacalcio.project.project.Graphic import Graphic
from fantacalcio.project.project.PredictiveOperation import PredictiveOperation
from fantacalcio.project.modelmissingvalues.SVRRegression import SVRRegression
from fantacalcio.project.database.SqlLiteDatabase import SqlLiteDatabase
from fantacalcio.common.ThreadPool import ThreadPool


def __define_dataframe_for_year__(votes, years, days):
    return pd.DataFrame({'vote': pd.Series(votes, dtype='float'),
                         'year': pd.Series(years, dtype='str'),
                         'day': pd.Series(days, dtype='int')})


class PlayerOperation:
    __total_day__: int = 38

    def __init__(self, path, name_table, name_player, name_team):
        self.__all_model__ = ['Sarima', 'Arima', 'MLP', 'LSTM', 'Keras']
        self.__sql_operation = SqlLiteDatabase(path, name_table)
        self.__name_player__ = name_player
        self.__name_team__ = name_team
        self.__graphic__ = Graphic()
        self.__svr__ = SVRRegression()
        self.__predictive_operation = PredictiveOperation()

    def start(self):
        self.__sql_operation.start()
        if self.__name_player__ is not None:
            self.__select_info_player__()
        elif self.__name_team__ is not None:
            self.__select_info_team__()
        else:
            self.__select_all_information__()

    def info_name_player(self, limit):
        self.__sql_operation.start()
        for name_player in self.__sql_operation.select_name_player(limit):
            print(name_player)

    def __select_info_player__(self):
        info_player = np.empty((0, 3))
        for vote, day, year in self.__sql_operation.select_info_player(self.__name_player__):
            info = np.array([[vote, year, day]])
            info_player = np.append(info_player, info, axis=0)
        # self.__graphic__.plot_figure(info_player[:, 0], info_player[:, 1], self.__name_player__)
        new_info_with_fill_hole = self.__insert_white_space_in_missing_values(info_player)
        # self.__graphic__.plot_figure_by_year(new_info_with_fill_hole[1:, 0], info_player[:, 1], self.__name_player__)
        new_info_without_fill_hole = self.__fill_nan_values_with_df_method__(pd.Series(new_info_with_fill_hole[1:, 0]))
        # self.__graphic__.plot_figure_by_year(new_info_without_fill_hole, info_player[:, 1], self.__name_player__)
        self.__predictive_operation.start(new_info_without_fill_hole, self.__name_player__)

    def __select_info_team__(self):
        for name, vote, day, year in self.__sql_operation.select_info_team(self.__name_team__):
            print(f'team {self.__name_team__} name {name} fanta vote {vote} day {day} year {year}')

    def __insert_white_space_in_missing_values(self, info_player):
        votes = np.empty(38)
        years = np.empty(38)
        days = np.empty(38)
        votes[:] = np.NaN
        years[:] = np.NaN
        days[:] = np.NaN
        default_dataframe = __define_dataframe_for_year__(votes, years, days).set_index(
            [pd.Index([index for index in range(1, self.__total_day__ + 1)])])
        final_values = [['vote', 'year', 'day']]
        for year in np.unique(info_player[:, 1]):
            filter_numpy = np.array(list(filter(lambda date: date[1] == year, info_player)))
            final_values_aux = self.__fill_hole__(default_dataframe.copy(), filter_numpy)
            final_values = np.append(final_values, final_values_aux, axis=0)
        return final_values

    @staticmethod
    def __replace__(array_player):
        return np.where(array_player == '-', np.nan, array_player)

    def __fill_hole__(self, default_df, actual_df):
        final_df = __define_dataframe_for_year__(self.__replace__(actual_df[:, 0]).astype(float),
                                                 self.__replace__(actual_df[:, 1]).astype(str),
                                                 self.__replace__(actual_df[:, 2]).astype(int))

        final_df = final_df.set_index([pd.Index([value[2] for value in actual_df])])
        for index in final_df.index:
            default_df.loc[int(index), :] = final_df.loc[index, :]
        return default_df.values

    def __continuation_operation__(self, new_info_without_fill_hole, name_player) -> list:
        all_process = []
        for model in self.__all_model__:
            all_process.append(ThreadPool.get_pool().apply_async(self.__predictive_operation.start,
                                                                 (new_info_without_fill_hole, name_player, model)))
        return all_process

    @staticmethod
    def __fill_nan_values_with_df_method__(series_player):
        return series_player.fillna(method='ffill').values.reshape(-1, 1)

    def __select_all_information__(self):
        all_player = self.__sql_operation.read_player_in_this_session()
        player_with_min_vote = self.__sql_operation.read_player_with_min_votes()
        all_player.day = pd.to_datetime(all_player.day + '0', format='%Y-%W%w', errors='raise')
        all_player = all_player.set_index('day')
        self.__sharpe_calculus__(all_player, player_with_min_vote)
        del self.__sql_operation
        self.__start_all_process__(all_player, player_with_min_vote)

    def __start_all_process__(self, all_player, player_with_min_vote):
        all_process = self.__start_process_filling_data__(all_player, player_with_min_vote)
        all_process_model = []
        for process, name_player in all_process:
            process.wait()
            all_process_model.append(self.__continuation_operation__(process.get(), name_player))
        [final_process.wait() for process_for_player in all_process_model for final_process in process_for_player]
        print(123)

    @staticmethod
    def __sharpe_calculus__(all_player, player_with_min_vote):
        f_write_descriptions = False
        if f_write_descriptions:
            df_mean = pd.DataFrame(index=np.arange(0, len(player_with_min_vote)),
                                   columns=['id', 'name', 'num', 'voto', 'stdev', 'sharpe'])
            for i_player in range(len(player_with_min_vote)):
                player_with_min_vote = all_player.query(
                    "name_player == '" + player_with_min_vote.iloc[i_player]["name_player"].replace("'", "\\'") + "'")
                player_with_min_vote = player_with_min_vote[player_with_min_vote.fanta_voto != '-']
                mean = player_with_min_vote.fanta_voto.mean()
                st_dev = player_with_min_vote.fanta_voto.std()
                sharpe = (mean - 6) / st_dev
                df_mean.loc[i_player] = [i_player, player_with_min_vote.iloc[i_player]["name_player"],
                                         len(player_with_min_vote), mean, st_dev, sharpe]

            df_mean.to_csv("averages.csv")

    def __start_process_filling_data__(self, all_player, player_with_min_vote):
        all_process = []
        for index in player_with_min_vote.index:
            name_player = player_with_min_vote.iloc[index]["name_player"]
            info_player = all_player.query(f'name_player == "{name_player}"')
            all_process.append(
                (ThreadPool.get_pool().apply_async(self.__filling_data__, (info_player,), ), name_player))
        return all_process

    def __filling_data__(self, info_player):
        # generate calendar
        idx = []
        for i in range(2014, 2021):
            for j in range(1, 39):
                idx.append("{0}-{1}".format(i, j))
        ds_idx = pd.Series(idx, index=idx, name='idx')
        ds_idx.index = pd.to_datetime(ds_idx.index + '0', format='%Y-%W%w', errors='raise')

        # align the series to calendar
        info_player = info_player.merge(ds_idx, left_index=True, right_index=True, how='outer')

        # fill the gaps by svr
        y = np.array(info_player.fanta_voto.replace("-", np.NaN), dtype=float)
        first, last, y_fill = self.__svr__.start(np.array(idx), y)
        # mavg = pd.Series(y_fill).rolling(window=4).mean()
        return y_fill.reshape(-1, 1)
