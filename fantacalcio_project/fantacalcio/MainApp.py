from project.project.PlayerOperation import PlayerOperation
from fantacalcio.common.ThreadPool import ThreadPool
from fantacalcio.project.project.VotePicker import VotePicker
import argparse


class MainApp:

    def __init__(self, file_giocatori: str, session: str, db_path: str, name_table: str, name_player: str,
                 name_team: str):
        super().__init__()
        self.__vote_picker__ = VotePicker(file_giocatori, session, db_path, name_table)
        self.__player_operation = PlayerOperation(db_path, name_table, name_player, name_team)

    def start(self):
        self.__vote_picker__.start()
        self.__player_operation.start()
        ThreadPool.close()
        ThreadPool.join()

    def info(self, limit: int):
        self.__player_operation.info_name_player(limit)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='download time serie fantacalcio.')
    parser.add_argument('session', metavar='session', type=str,
                        help='session to download in way "2018-2019"')
    parser.add_argument('db_path', metavar='db_path', type=str,
                        help='path for database, in this case sqllite database')
    parser.add_argument('name_table', metavar='name_table', type=str,
                        help='name table in database')
    parser.add_argument('--name_player', metavar='name_player', type=str,
                        help='name player in database')
    parser.add_argument('--name_team', metavar='name_team', type=str,
                        help='name team in database')
    parser.add_argument('--name_players', metavar='name_players', type=int,
                        help='random name player')

    args = parser.parse_args()
    print(f' init download for session {args.session}')

    MainApp('lista_giocatori.csv', args.session, args.db_path, args.name_table, args.name_player, args.name_team) \
        .info(args.name_players) if args.name_players is not None else MainApp('lista_giocatori.csv', args.session,
                                                                               args.db_path, args.name_table,
                                                                               args.name_player,
                                                                               args.name_team).start()
