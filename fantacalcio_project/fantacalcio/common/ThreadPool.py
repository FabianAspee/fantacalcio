from multiprocessing import Pool
import os


class ThreadPool:
    __instance = None

    @staticmethod
    def get_instance():
        if ThreadPool.__instance is None:
            ThreadPool()
        return ThreadPool.__instance

    def __init__(self):
        """ Virtually private constructor. """
        self.__pool__ = Pool(os.cpu_count())
        if ThreadPool.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            ThreadPool.__instance = self

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['__pool__']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)

    @staticmethod
    def get_pool():
        return ThreadPool.get_instance().__pool__

    @staticmethod
    def close():
        ThreadPool.__instance.__pool__.close()

    @staticmethod
    def join():
        ThreadPool.__instance.__pool__.join()
