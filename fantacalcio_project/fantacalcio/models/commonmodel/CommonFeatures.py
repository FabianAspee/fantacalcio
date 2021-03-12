from joblib import dump, load

from common.CommonFunction import CommonFunction


class CommonFeatures:

    @staticmethod
    def load_model(name_model):
        return load(CommonFunction.total_path(CommonFunction.base_path, f'../{name_model}.joblib'))

    @staticmethod
    def save_model(model, name_model):
        dump(model, CommonFunction.total_path(CommonFunction.base_path, f'../{name_model}.joblib'))