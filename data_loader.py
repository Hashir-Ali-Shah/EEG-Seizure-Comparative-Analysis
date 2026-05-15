import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
import os

class SeizureDataLoader:
    def __init__(self, base_path):
        self.base_path = base_path

    def load_beed(self):
        """Loads the BEED Bangalore EEG Epilepsy Dataset."""
        path = os.path.join(self.base_path, "BEED Bangalore EEG Epilepsy Dataset", "BEED_Data.csv")
        df = pd.read_csv(path)
        X = df.drop(columns=['y']).values
        # Map labels: 0 remains 0, any > 0 becomes 1
        y = (df['y'] > 0).astype(int).values
        return X, y

    def load_recognition(self):
        """Loads the Epileptic Seizure Recognition Dataset."""
        path = os.path.join(self.base_path, "Epileptic_Seizure_Recognition", "Epileptic Seizure Recognition.csv")
        df = pd.read_csv(path)
        # Drop the ID column
        X = df.drop(columns=['Unnamed', 'y']).values
        # Standardize labels: 1 is seizure, 2,3,4,5 are non-seizure
        # Map to binary: 1 -> 1, others -> 0
        y = (df['y'] == 1).astype(int).values
        return X, y

    def load_analysis(self, extract_features=True):
        """Loads the EEG Seizure Analysis Dataset (.npz)."""
        path = os.path.join(self.base_path, "EEG Seizure Analysis Dataset", "eeg-predictive_train.npz")
        data = np.load(path)
        signals = data['train_signals'] # Shape: (samples, channels, time_points)
        labels = data['train_labels']
        
        if extract_features:
            # Extract statistical features per channel
            # Shape (samples, channels, time_points) -> (samples, channels, num_features)
            mean = np.mean(signals, axis=2)
            std = np.std(signals, axis=2)
            max_val = np.max(signals, axis=2)
            min_val = np.min(signals, axis=2)
            sk = skew(signals, axis=2)
            kt = kurtosis(signals, axis=2)
            
            # Concatenate features: Resulting shape (samples, channels * 6)
            X = np.concatenate([mean, std, max_val, min_val, sk, kt], axis=1)
        else:
            # Flatten: (samples, channels * time_points)
            X = signals.reshape(signals.shape[0], -1)
            
        return X, labels

if __name__ == "__main__":
    # Test loading
    loader = SeizureDataLoader(r"d:\adnan_amin_project")
    
    print("Loading BEED...")
    X1, y1 = loader.load_beed()
    print(f"BEED Shape: {X1.shape}, Class Distribution: {np.bincount(y1)}")
    
    print("\nLoading Recognition...")
    X2, y2 = loader.load_recognition()
    print(f"Recognition Shape: {X2.shape}, Class Distribution: {np.bincount(y2)}")
    
    print("\nLoading Analysis (with feature extraction)...")
    X3, y3 = loader.load_analysis()
    print(f"Analysis Shape: {X3.shape}, Class Distribution: {np.bincount(y3)}")
