import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import BallTree

class MeanSqmPrice(BaseEstimator, TransformerMixin):
    def __init__(self, prices, total_areas):
        self.prices = prices
        self.total_areas = total_areas

    def fit(self, X, y = None):
        self.X_train = X.copy()

        self.tree = BallTree(X[:, [12, 13]])
        dists, indeces = self.tree.query(X[:, [12, 13]], k=5)
        mean_sqm_price_of_cluster = []
        for rows in indeces:
            mean_sqm_price_of_cluster.append(np.mean(self.X_train[rows, 0] / self.X_train[rows, 1]))
        X[:, -1] = mean_sqm_price_of_cluster

        X = X[:, 1:] # Drop price before fitting estimator
        return self

    def transform(self, X, y = None):
        dists, indeces = self.tree.query(X[:, [12, 13]], k=5)
        mean_sqm_price_of_cluster = []

        for rows in indeces:
            mean_sqm_price_of_cluster.append(np.mean(self.prices[rows] / self.total_areas[rows]))

        X = np.c_[X, mean_sqm_price_of_cluster]
        
        X = X[:, 1:] # Drop price before making prediction

        return X