import sklearn
import sklearn.model_selection
import numpy as np
from data_split import DataSplit

class SklearnOptimiser():

    classifier: sklearn.base.BaseEstimator

    optimalClassifier: sklearn.base.BaseEstimator

    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray

    X_full: np.ndarray
    y_full: np.ndarray

    parameterGrid: list[dict[str, object]]



    scores: list[tuple]

    def __init__(self, data: DataSplit, classifier: sklearn.base.BaseEstimator, parameterGrid: list[dict[str, object]]):

        self.X_train, self.y_train = data.train.to_numpy(flatten = True)
        self.X_test, self.y_test = data.test.to_numpy(flatten = True)
        self.X_val, self.y_val = data.val.to_numpy(flatten = True)
        self.classifier = classifier
        self.parameterGrid = parameterGrid
        self.X_full = np.concatenate((self.X_train, self.X_test), axis=0)
        self.y_full = np.concatenate((self.y_train, self.y_test), axis=0)

    def optimize(self, optimizer: str, n_jobs = 1):

        if optimizer == "grid search" or optimizer == "grid_search" or optimizer == "gridsearch":
            self.grid = sklearn.model_selection.GridSearchCV(self.classifier, self.parameterGrid, n_jobs=n_jobs)
            self.grid.fit(self.X_full, self.y_full)
            self.optimalClassifier = self.grid.best_estimator_

        else:
            raise ValueError("Unsupported optimizer type: " + optimizer)
        
    def getOptimalClassifier(self) -> sklearn.base.BaseEstimator:
        return self.optimalClassifier
    
    def getScores(self) -> list[tuple]:
        return self.scores
    
    def getOptimalParameters(self) -> dict[str, object]:
        return self.optimalClassifier.get_params()
