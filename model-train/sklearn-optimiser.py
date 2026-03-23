import sklearn
import sklearn.model_selection
import numpy as np
from data_split import DataSplit

class SklearnOptimiser():

    classifierType: sklearn.base.BaseEstimator

    optimalClassifier: sklearn.base.BaseEstimator

    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray

    parameterGrid: list[dict[str, object]]

    scores: list[tuple]

    def __init__(self, data: DataSplit, classifierType: sklearn.base.BaseEstimator, parameterGrid: list[dict[str, object]]):

        self.X_train, self.y_train = data.train.to_numpy(flatten = True)
        self.X_test, self.y_test = data.test.to_numpy(flatten = True)
        self.X_val, self.y_val = data.val.to_numpy(flatten = True)
        self.classifierType = classifierType
        self.parameterGrid = parameterGrid

    def optimize(self, optimizer: str):

        if optimizer == "grid search":
            sklearn.model_selection.GridSearchCV(self.classifierType, self.parameterGrid)
