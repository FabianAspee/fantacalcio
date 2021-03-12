import json
import os
import sqlite3
from sqlite3 import Error
import numpy as np
import pandas as pd
from pandas import DataFrame
from sqlalchemy import create_engine
from common.CommonFunction import CommonFunction


def __query_for_check_table__(table_name) -> str:
    return f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';"


def __select_random_element__(limit) -> str:
    return f"SELECT name_player FROM player_info ORDER BY random() LIMIT {limit};"


def __query_for_insert__() -> str:
    return ''' INSERT INTO player_info(teams_name,name_player,player_position,vote,goal,assistance,penalties_scored_saved,
     missed_penalties,own_goal,admonitions,expulsion,fanta_voto,year,day) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?) '''


def __query_for_select__(limit: int) -> str:
    return f'SELECT * FROM player_info LIMIT {limit}'


def __select_story_series_player__(name_player: str) -> str:
    return f'SELECT fanta_voto, day, year FROM player_info WHERE name_player = "{name_player}"'


def __select_player_play_in_this_session__() -> str:
    return "SELECT id, teams_name, name_player, player_position, vote, fanta_voto, substr(year,1,4)||'-'||day as day " \
           "FROM player_info ORDER BY name_player "


def __select_player_with_min_vote__(min_votes: int) -> str:
    return f'SELECT name_player,count() as cnt FROM player_info WHERE name_player in ( SELECT name_player FROM ' \
           f'player_info WHERE year = "2020-20" GROUP BY name_player) group by name_player HAVING' \
           f' cnt > {str(min_votes)} ORDER BY cnt DESC'


def __select_story_series_team__(teams_name: str) -> str:
    return f'SELECT name_player, fanta_voto, day, year FROM player_info WHERE teams_name = "{teams_name}" ORDER BY ' \
           f'name_player, year '


def __query_for_verify_year__(year_init: str) -> str:
    return f'SELECT COUNT(year) FROM player_info WHERE year =  "{year_init}"'


def __create_connection__(db_file):
    try:
        return sqlite3.connect(db_file)
    except Error as e:
        print(e)


class SqlLiteDatabase:
    __path_schema__ = '../resources/schema_database.json'
    __path_db__ = '../resources/fanta.db'

    def __init__(self, db_file, name_table):
        self.__connection__ = None
        self.__db_path__ = db_file
        self.__name_table__ = name_table

    def start(self):
        self.__connection__ = __create_connection__(
            CommonFunction.total_path(CommonFunction.base_path, f'../{self.__db_path__}'))
        if self.__check_table_player__():
            self.__create_table__(self.__read_json_file_table__())

    def select_name_player(self, limit):
        result = self.__connection__.cursor().execute(__select_random_element__(limit))
        rows = result.fetchall()
        for name in rows:
            yield name[0]

    def years_exists_in_database(self, years) -> (list, bool):
        result = list(map(lambda year: (year, True) if self.__connection__.cursor()
                          .execute(__query_for_verify_year__(year))  # conrollar
                          .fetchall()[0][0] != 0 else (year, False), years))
        condition = all(condition is True for (_, condition) in result)
        return list(
            map(lambda year: year[0], filter(lambda filter_element: filter_element[1] is False, result))), condition

    def __check_table_player__(self) -> bool:
        result = self.__connection__.cursor().execute(__query_for_check_table__(self.__name_table__))
        rows = result.fetchall()
        return True if len(rows) == 0 else False

    def insert_info_player(self, info_player: np.array):
        for values in info_player[1:, :]:
            sql = __query_for_insert__()
            cur = self.__connection__.cursor()
            cur.execute(sql, values)
        self.__connection__.commit()

    def select_first_n_esimo_player(self, n_player):
        result = self.__connection__.cursor().execute(__query_for_select__(n_player))
        rows = result.fetchall()
        for result in rows:
            print(result)

    def select_info_player(self, name_player):
        result = self.__connection__.cursor().execute(__select_story_series_player__(name_player))
        rows = result.fetchall()
        for result in rows:
            yield result

    def select_info_team(self, name_team):
        result = self.__connection__.cursor().execute(__select_story_series_team__(name_team))
        rows = result.fetchall()
        for result in rows:
            yield result

    def close_connection(self):
        if self.__connection__:
            self.__connection__.close()

    def __read_json_file_table__(self):
        with open(CommonFunction.total_path(CommonFunction.base_path, self.__path_schema__)) as json_file:
            return json.load(json_file)['table_player']

    def __create_table__(self, query_create_table):
        try:
            self.__connection__.cursor().execute(query_create_table)
        except Error as e:
            print(e)

    def __read_sql_lite_with_sql_alchemy__(self, query, by_record=False):
        engine = create_engine(f'sqlite:///{CommonFunction.total_path(CommonFunction.base_path, self.__path_db__)}')
        with engine.connect() as connection:
            if by_record:
                try:
                    result = connection.execute(query)
                except Exception as e:
                    print(e)
                else:
                    result.close()
            else:
                query_df = pd.read_sql(query, con=engine)
        return query_df

    def read_player_in_this_session(self) -> DataFrame:
        return self.__read_sql_lite_with_sql_alchemy__(__select_player_play_in_this_session__())

    def read_player_with_min_votes(self, min_vote=10) -> DataFrame:
        return self.__read_sql_lite_with_sql_alchemy__(__select_player_with_min_vote__(min_vote))

    def save_prediction_and_std_for_player(self, prediction, std):
        pass