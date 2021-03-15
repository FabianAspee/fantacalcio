from abc import ABC, abstractmethod

from common.CommonFunction import CommonFunction


class GeneralFunction(ABC):

    @abstractmethod
    def create_variable_for_model(self, player, windows_size=13):
        windows, player_vote = CommonFunction.compute_windows(player, windows_size)
        x, y = windows[:int(len(windows) * 0.8)], windows[int(len(windows) * 0.8):]
        return x, player_vote[:int(len(player_vote) * 0.8)], y, player_vote[
                                                                int(len(player_vote) * 0.8):]
