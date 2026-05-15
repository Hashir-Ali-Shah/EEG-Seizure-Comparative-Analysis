from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class OutlierClipper(BaseEstimator, TransformerMixin):
    """Simple noise removal by clipping outliers beyond N standard deviations."""
    def __init__(self, sigma=3):
        self.sigma = sigma
        self.upper_ = None
        self.lower_ = None

    def fit(self, X, y=None):
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        self.upper_ = mean + self.sigma * std
        self.lower_ = mean - self.sigma * std
        return self

    def transform(self, X):
        return np.clip(X, self.lower_, self.upper_)

def get_pipeline_a(n_features=10):
    """Pipeline A: Normalization -> Noise removal -> Feature selection"""
    return Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('noise_removal', OutlierClipper(sigma=3)),
        ('feature_selection', SelectKBest(score_func=f_classif, k=n_features))
    ])

def get_pipeline_b(n_components=0.95):
    """Pipeline B: Scaling -> PCA"""
    return Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MinMaxScaler()),
        ('pca', PCA(n_components=n_components))
    ])
