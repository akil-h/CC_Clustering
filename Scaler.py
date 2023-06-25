from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted
import numpy as np

class CustomerScaler(TransformerMixin, BaseEstimator):
    def __init__(self, with_mean = True):
        self.with_mean = with_mean

    def fit(self, X, y=None):
        X = check_array(X)