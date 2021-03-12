from fantacalcio.common.ThreadPool import ThreadPool
from fantacalcio.project.project.PredictiveOperation import PredictiveOperation


class ProcessPlayer:
    __instance = None

    @staticmethod
    def get_instance():
        if ProcessPlayer.__instance is None:
            ProcessPlayer()
        return ProcessPlayer.__instance

    def __init__(self):
        """ Virtually private constructor. """
        self.__player_info__ = [()]
        self.__predictive_operation = PredictiveOperation()
        if ProcessPlayer.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            ProcessPlayer.__instance = self

    @staticmethod
    def append_player(info_player, name_player):
        ProcessPlayer.get_instance().__player_info__.append((info_player, name_player))

    @staticmethod
    def process_all_player_in_models(model):
        all_process = []
        for new_info_without_fill_hole, name_player in ProcessPlayer.get_instance().__player_info__:
            all_process.append(ThreadPool.get_pool().apply_async(ProcessPlayer.get_instance().__predictive_operation.
                               start, (new_info_without_fill_hole, name_player, model)))
        return all_process
