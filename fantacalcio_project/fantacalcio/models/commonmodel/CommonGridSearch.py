from sklearn.model_selection import GridSearchCV


class CommonGridSearch:

    @staticmethod
    def __create_grid_search__(model, parameters, data, target):
        return GridSearchCV(model, parameters).fit(data, target)

    @staticmethod
    def get_best_index(model, parameters, data, target) -> int:
        return CommonGridSearch.__create_grid_search__(model, parameters, data, target).best_index_()

    @staticmethod
    def get_all_result(model, parameters, data, target) -> dict:
        return CommonGridSearch.__create_grid_search__(model, parameters, data, target).cv_results_

    @staticmethod
    def get_best_params(model, parameters, data, target) -> dict:
        return CommonGridSearch.__create_grid_search__(model, parameters, data, target).best_params_
