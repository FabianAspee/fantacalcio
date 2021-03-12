from dataclasses import dataclass
from bs4 import ResultSet


@dataclass(frozen=True)
class InfoPlayer:
    init_index: int
    finish_index: int
    team_name: ResultSet
    name_players: ResultSet
    player_roles: list
    vote: [[]]
    another_parameter: [[]]
    fanta_voto: [[]]
