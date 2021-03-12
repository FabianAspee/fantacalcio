from urllib.request import urlopen, Request
import pandas as pd
import requests
import bs4

from common.ThreadPool import ThreadPool
from caseclasspython.InfoPlayers import InfoPlayer
from project.project.Player import Player
from project.database.SqlLiteDatabase import SqlLiteDatabase
import numpy as np


def __get_vote__(player, vote_for_player):
    return not vote_for_player['Info'][vote_for_player['Info']['Gio'] == player.get_name()].empty


def __filter_function__(tag, my_attr, text_tag):
    if not isinstance(tag, bs4.NavigableString):
        new_value = list(
            filter(lambda element: not isinstance(element, bs4.NavigableString) and element.has_attr("class")
                   and element.get("class") == my_attr and element.text != text_tag, tag.children))
        return new_value


def __filter_function__list__(tag, my_attr, text_tag):
    if not isinstance(tag, bs4.NavigableString):
        new_value = list(
            filter(lambda element: not isinstance(element, bs4.NavigableString) and element.has_attr("class")
                   and element.get("class") == my_attr and element.text not in text_tag, tag.children))
        return new_value


def __filter_vote__(value):
    return list(filter(lambda final_value: final_value is not None and len(final_value) > 0,
                       map(lambda element: __filter_function__(element, ['inParameter', 'vParameter'], 'V'), value)))


def __filter_another_parameter__(value):
    return list(filter(lambda final_value: final_value is not None and len(final_value) > 0,
                       map(lambda element: __filter_function__list__(element, ['inParameter'],
                                                                     ['G', 'A', 'R', 'AM', 'RS', 'AG',
                                                                      'ES']), value)))


def __filter_fanta_voto__(value):
    return list(filter(lambda final_value: final_value is not None and len(final_value) > 0,
                       map(lambda element: __filter_function__(element, ['inParameter', 'fvParameter'],
                                                               'FV'), value)))


class VotePicker:
    __day__ = 39

    def __init__(self, file_giocatori, years, path, name_table):
        super().__init__()
        self.__player_list__ = []
        self.__file_giocatori__: str = file_giocatori
        self.__years__ = self.__create_array_session(years)
        self.__path__: str = path
        self.__name_table__: str = name_table

    def start(self):
        connection = SqlLiteDatabase(self.__path__, self.__name_table__)
        connection.start()
        self.__years__, condition = connection.years_exists_in_database(self.__years__)
        if condition is False:
            for total_info_web_site, year, day in self.__get_information_from_site__gazzetta__(self.__years__):
                print(f'process year {year} in day {day}')
                connection.insert_info_player(total_info_web_site)

        connection.select_first_n_esimo_player(5)

    def __initialized_list__(self):
        self.__vote_for_match_day__ = []
        self.__match_days__ = []

    def __make_vote_series__(self, info_votes):
        for player in self.__player_list__:
            self.__initialized_list__()
            if __get_vote__(player, info_votes):  # fare
                print(f"Serie voti di {player.get_name()} creata:")
                self.__vote_for_match_day__.sort(key=lambda tup: tup[0])  # sorts in place
                for pair in self.__vote_for_match_day__:
                    player.add_vote(pair)
                player.take_sorted_vote_for_match_day(self.__vote_for_match_day__)

    def __get_player_list__(self):
        player_list = []
        info_player = pd.read_csv(f'resources\\{self.__file_giocatori__}')
        for _, row in info_player.iterrows():
            player_list.append(Player(row['R'], row['Nome'], row['Squadra'], row['Qt. I']))

        return player_list

    @staticmethod
    def __get_information_from_site__(period):
        for year in period:
            print(year)
            url = f"https://www.fantapiu3.com/fantacalcio-storico-voti-fantapiu3-{year}.php"
            headers = {
                'User-Agent': 'whatever'}  # The server seems fussy about which user agent is accessing the resource
            request = Request(url, headers=headers)
            page = urlopen(request)
            html_bytes = page.read()
            html = html_bytes.decode("utf-8")
            start_index = html.find('<table class="table table-hover game-player-result">')
            end_index = html.find("</table>") + len("</table>")
            table = html[start_index:end_index]
            yield pd.read_html(table)[0], year

    def __start_download_information__(self, day, year, info_teams):
        url = f"https://www.gazzetta.it/calcio/fantanews/voti/serie-a-{year}/giornata-{day}"
        headers = {
            'User-Agent': 'whatever'}  # The server seems fussy about which user agent is accessing the resource
        page = requests.get(url, headers=headers)
        soup = bs4.BeautifulSoup(page.content, 'html.parser')
        teams_limit = self.__find_all_teams__(soup)
        teams = soup.find_all('ul', class_='magicTeamList', limit=teams_limit)
        for value in teams:
            info_teams = self.__fill_info_teams__(value, info_teams, year, day)
        return info_teams, year, day

    def __get_information_from_site__gazzetta__(self, period):
        all_process = []
        for year in period:
            info_teams = np.array([['teams_name', 'name_player', 'player_position', 'vote', 'goal',
                                    'assistance', 'penalties_scored_saved', 'missed_penalties',
                                    'own_goal', 'admonitions', 'expulsion', 'fanta_voto', 'year', 'day']])
            for day in range(1, self.__day__):
                all_process.append(
                    ThreadPool.get_pool().apply_async(self.__start_download_information__, (day, year, info_teams), ))

        for process in all_process:
            process.wait()
            info_teams, year, day = process.get()
            print(f'total info teams year {year} in day {day} len {len(info_teams)} ')
            yield info_teams, year, day

    @staticmethod
    def __find_all_teams__(soup):
        count = 0
        for value in soup.find('ul', class_='menuTeams').children:
            if not isinstance(value, bs4.NavigableString):
                count += 1
        return count

    def __fill_info_teams__(self, value, info_teams, year, day):
        name_players = value.find_all('a', {'target': '_blank'})
        player_roles = value.find_all('span', {'class': 'playerRole'})
        team_name = value.find_all('span', {'class': 'teamNameIn'})
        vote = __filter_vote__(value)
        another_parameter = __filter_another_parameter__(value)
        fanta_voto = __filter_fanta_voto__(value)

        init_index = len(info_teams)
        finish_index = len(name_players) + len(info_teams)
        info_team = InfoPlayer(init_index, finish_index, team_name, name_players, [player_roles[index]
                                                                                   for index in
                                                                                   range(0, len(player_roles), 2)],
                               vote,
                               another_parameter, fanta_voto)
        return self.__insert_info_players__(info_teams, info_team, year, day)

    @staticmethod
    def __insert_info_players__(info_teams, info_player, year, day):
        flatten_vote = [vote for value in info_player.vote for vote in value]
        flatten_fanta_voto = [fanta_vote for value in info_player.fanta_voto for fanta_vote in value]
        team_name = [value.text for value in info_player.team_name][0]
        for name_players, player_roles, vote, another_parameter, fanta_voto in zip(info_player.name_players,
                                                                                   info_player.player_roles,
                                                                                   flatten_vote,
                                                                                   info_player.another_parameter,
                                                                                   flatten_fanta_voto):
            info_various = np.array(
                [[team_name, name_players.text, player_roles.text, vote.text, another_parameter[0].text,
                  another_parameter[1].text, another_parameter[2].text, another_parameter[3].text,
                  another_parameter[4].text, another_parameter[5].text, another_parameter[6].text,
                  fanta_voto.text, year, day]])

            info_teams = np.append(info_teams, info_various, axis=0)
        return info_teams

    @staticmethod
    def __create_array_session(years):
        session = []
        first_year = years[0:4]
        len_year = len(years)
        end_year = years[len_year - 4:len_year]
        first_year_int = int(first_year)
        end_year_int = int(end_year)
        for year in range(1 + end_year_int - first_year_int):
            session.append(f'{first_year_int}-{end_year[:2]}')
            first_year_int = first_year_int + 1
        return session

    def get_player_list(self):
        return self.__player_list__

    def set_player_list(self, player_list):
        self.__player_list__ = player_list
