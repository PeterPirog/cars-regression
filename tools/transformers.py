# https://github.com/scikit-learn-contrib/category_encoders/blob/master/category_encoders/one_hot.py
# pip install category_encoders
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa
# now you can import normally from sklearn.impute
from sklearn.impute import IterativeImputer as _IterativeImputer
from sklearn.preprocessing import QuantileTransformer as _QuantileTransformer
from sklearn.impute import SimpleImputer as _SimpleImputer

# https://github.com/scikit-learn/scikit-learn/issues/5523
def check_output(X, ensure_index=None, ensure_columns=None):
    """
    Joins X with ensure_index's index or ensure_columns's columns when avaialble
    """
    if ensure_index is not None:
        if ensure_columns is not None:
            if type(ensure_index) is pd.DataFrame and type(ensure_columns) is pd.DataFrame:
                X = pd.DataFrame(X, index=ensure_index.index, columns=ensure_columns.columns)
        else:
            if type(ensure_index) is pd.DataFrame:
                X = pd.DataFrame(X, index=ensure_index.index)
    return X


class IterativeImputer(_IterativeImputer):
    def transform(self, X):
        Xt = super(IterativeImputer, self).transform(X)
        return check_output(Xt, ensure_index=X, ensure_columns=X)

    def fit_transform(self, X, y):
        Xt = super(IterativeImputer, self).fit_transform(X)
        return check_output(Xt, ensure_index=X, ensure_columns=X)


class QuantileTransformer(_QuantileTransformer):
    def transform(self, X):
        Xt = super(QuantileTransformer, self).transform(X)
        return check_output(Xt, ensure_index=X, ensure_columns=X)

    def fit_transform(self, X, y):
        Xt = super(QuantileTransformer, self).fit_transform(X)
        return check_output(Xt, ensure_index=X, ensure_columns=X)


class SimpleImputer(_SimpleImputer):
    def transform(self, X):
        Xt = super(SimpleImputer, self).transform(X)
        return check_output(Xt, ensure_index=X, ensure_columns=X)

    def fit_transform(self, X, y):
        Xt = super(SimpleImputer, self).fit_transform(X)
        return check_output(Xt, ensure_index=X, ensure_columns=X)