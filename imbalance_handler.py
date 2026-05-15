from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import numpy as np

def handle_imbalance(X, y, method='smote'):
    """Applies SMOTE or Undersampling to the dataset."""
    if method == 'smote':
        sampler = SMOTE(random_state=42)
    elif method == 'undersample':
        sampler = RandomUnderSampler(random_state=42)
    else:
        return X, y
        
    X_res, y_res = sampler.fit_resample(X, y)
    return X_res, y_res
