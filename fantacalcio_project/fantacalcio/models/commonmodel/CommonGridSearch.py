from sklearn.model_selection import GridSearchCV


class CommonGridSearch:

    @staticmethod
    def __create_grid_search__(model, parameters, data, target, num_processor):
        return GridSearchCV(model, parameters, cv=10, n_jobs=num_processor).fit(data, target)

    @staticmethod
    def get_best_index(model, parameters, data, target) -> int:
        return CommonGridSearch.__create_grid_search__(model, parameters, data, target).best_index_()

    @staticmethod
    def get_all_result(model, parameters, data, target) -> dict:
        return CommonGridSearch.__create_grid_search__(model, parameters, data, target).cv_results_

    @staticmethod
    def get_best_params(model, parameters, data, target, num_processor) -> dict:
        return CommonGridSearch.__create_grid_search__(model, parameters, data, target, num_processor).best_params_
