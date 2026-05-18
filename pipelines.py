from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.stats import skew, kurtosis

class EEGBandpassFilter(BaseEstimator, TransformerMixin):
    """Applies a Butterworth bandpass filter to EEG raw signals."""
    def __init__(self, lowcut=0.5, highcut=40.0, fs=100.0, order=4):
        self.lowcut = lowcut
        self.highcut = highcut
        self.fs = fs
        self.order = order
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        nyq = 0.5 * self.fs
        low = self.lowcut / nyq
        high = self.highcut / nyq
        b, a = butter(self.order, [low, high], btype='band')
        
        X_filtered = X.copy()
        if len(X.shape) == 3:
            # Apply filter along the time axis (axis=2) for 3D signals
            X_filtered = filtfilt(b, a, X_filtered, axis=2)
        elif len(X.shape) == 2:
            # Apply filter along the time axis (axis=1) for 2D signals
            X_filtered = filtfilt(b, a, X_filtered, axis=1)
        return X_filtered

class EEGFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extracts 6 statistical features (mean, std, max, min, skew, kurtosis) from EEG signals."""
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        # If X is 3D: (samples, channels, time_points)
        if len(X.shape) == 3:
            mean = np.mean(X, axis=2)
            std = np.std(X, axis=2)
            max_val = np.max(X, axis=2)
            min_val = np.min(X, axis=2)
            sk = skew(X, axis=2)
            kt = kurtosis(X, axis=2)
            return np.concatenate([mean, std, max_val, min_val, sk, kt], axis=1)
            
        # If X is 2D (single channel): (samples, time_points)
        elif len(X.shape) == 2:
            # Slicing into 5 sub-windows to extract temporal feature patterns
            n_windows = 5
            window_size = X.shape[1] // n_windows
            features = []
            for w in range(n_windows):
                window = X[:, w*window_size : (w+1)*window_size]
                features.append(np.mean(window, axis=1))
                features.append(np.std(window, axis=1))
                features.append(np.max(window, axis=1))
                features.append(np.min(window, axis=1))
                features.append(skew(window, axis=1))
                features.append(kurtosis(window, axis=1))
            return np.column_stack(features)
            
        return X

def get_pipeline_a(is_raw=True):
    """Pipeline A (Signal quality emphasis): Raw EEG -> bandpass filtering -> statistical feature extraction"""
    if is_raw:
        return Pipeline([
            ('bandpass', EEGBandpassFilter()),
            ('features', EEGFeatureExtractor()),
            ('imputer', SimpleImputer(strategy='mean'))
        ])
    else:
        # Pass-through for already pre-extracted tabular features
        return Pipeline([
            ('imputer', SimpleImputer(strategy='mean'))
        ])

def get_pipeline_b(is_raw=True, n_components=0.95):
    """Pipeline B (Representation + generalization): Raw EEG -> feature extraction -> scaling -> PCA"""
    if is_raw:
        return Pipeline([
            ('features', EEGFeatureExtractor()),
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=n_components, svd_solver='full'))
        ])
    else:
        # Pre-extracted tabular features -> scale and compress with PCA
        return Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=n_components, svd_solver='full'))
        ])
