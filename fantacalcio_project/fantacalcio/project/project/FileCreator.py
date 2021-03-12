import pandas as pd


class FileCreator:
    def __init__(self, player_list, year):
        super().__init__()
        self.__player_list__ = player_list
        self.__year__ = year

    def start(self):
        self.__index_voted_series__()

    def __index_voted_series__(self):
        final = pd.DataFrame(self.__player_list__)
        final.to_csv(f'resources/final_information-{self.__year__}.csv')
