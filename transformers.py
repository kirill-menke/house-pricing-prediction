import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class ReplaceCommaWithDot(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y = None):
        return self

    def transform(self, X, y = None):
        return X.replace(to_replace=r',', value='.', regex=True)