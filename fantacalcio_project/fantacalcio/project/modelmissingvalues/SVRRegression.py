import numpy as np
from sklearn.svm import SVR

from fantacalcio.caseclasspython.CaseSVRModel import CaseSVRModel
from fantacalcio.common.CommonFunction import CommonFunction


class SVRRegression:

    def start(self, x, y):
        x_p, y_p = self.__tick_with_non_nan__(x, y)
        x_s, model_x = CommonFunction.standardize(np.array(x_p).reshape(-1, 1))
        y_s, model_y = CommonFunction.standardize(np.array(y_p).reshape(-1, 1))
        svr_model = self.__fit__(self.__create_svr__(), x_s, y_s)
        first, last, yp = self.__calculate_missing_values__(
            CaseSVRModel(x_p, x_s, y_s, y_p, model_x, model_y, svr_model))
        return first, last, yp

    @staticmethod
    def __create_svr__():
        return SVR(kernel='rbf', C=100, gamma=10)

    @staticmethod
    def __fit__(regressor, x_s, y_s):
        return regressor.fit(x_s, y_s.reshape(-1, ))

    """
        # rbf: C trades off mis classification against simplicity of the decision surface.
        # low C: decision surface smooth, high C: classify all examples correctly.
        # gamma defines how how far the influence of a single training example reaches
        # The larger gamma, the closer other examples must be to be affected.


        # y_pred = sc_y.inverse_transform ((regressor.predict (sc_X.transform(np.array([[6.5]])))))
        #ypred = case_svr_model.model_y.inverse_transform((case_svr_model.svr_model.predict(case_svr_model.x_s)))
    """

    @staticmethod
    def __calculate_missing_values__(case_svr_model):
        first = case_svr_model.x_p[0]
        last = case_svr_model.x_p[len(case_svr_model.x_p) - 1]
        yp = case_svr_model.model_y.inverse_transform(
            (case_svr_model.svr_model.predict(case_svr_model.model_x.transform(np.arange(first, last).reshape(-1, 1)))))
        for i in range(first, last):
            for j in range(len(case_svr_model.x_p)):
                if i == case_svr_model.x_p[j]:
                    yp[i - first] = case_svr_model.y_p[j]

        return first, last, yp

    @staticmethod
    def __standardize__(x_train):
        return CommonFunction.standardize(x_train)

    @staticmethod
    def __tick_with_non_nan__(x, y):
        first = -1
        x_p = []  # ticks with non NaNs y
        y_p = []
        for i in range(len(x)):
            if not (np.isnan(y[i])):
                if first < 0:
                    first = i
                x_p.append(i)
                y_p.append(y[i])
        return x_p, y_p
