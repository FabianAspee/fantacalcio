from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


@dataclass(frozen=True)
class CaseSVRModel:
    x_p: list
    x_s: [[]]
    y_s: [[]]
    y_p: list
    model_x: StandardScaler
    model_y: StandardScaler
    svr_model: SVR
