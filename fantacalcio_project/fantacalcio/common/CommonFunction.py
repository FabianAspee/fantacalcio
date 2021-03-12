import os

from sklearn import preprocessing


class CommonFunction:
    base_path = os.path.dirname(__file__)
    total_path = os.path.join

    @staticmethod
    def standardize(x_train):
        standard = preprocessing.StandardScaler()
        return standard.fit_transform(x_train), standard

    @staticmethod
    def normalize(x_train):
        normalizer = preprocessing.Normalizer()
        return normalizer.fit_transform(x_train), normalizer

    @staticmethod
    def min_max_Scaler(x_train):
        min_max = preprocessing.MinMaxScaler()
        return min_max.fit_transform(x_train), min_max
