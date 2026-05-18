# Updated Preprocessing Pipelines

This document details the newly proposed preprocessing pipelines designed to evaluate noise reduction, information compression, and generalization performance on EEG datasets.

---

## Pipeline A: Signal Quality Emphasis

### 1. The Processing Flow
`Raw EEG → Bandpass Filtering → Statistical Feature Extraction → Logistic Regression`

### 2. Design Details & Core Tests
* **Noise Reduction**: Uses bandpass filtering to remove low-frequency drifts and high-frequency muscle noise from the raw continuous brainwaves.
* **Simple and Clean**: Features are extracted directly from the filtered wave and fed straight to the model with no extra scaling or modifications.
* **Key Test**: Designed to evaluate the exact impact of basic signal filtering on prediction performance.

---

## Pipeline B: Representation & Generalization

### 1. The Processing Flow
`Raw EEG → Statistical Feature Extraction → Scaling → PCA → Logistic Regression`

### 2. Design Details & Core Tests
* **ML-Focused**: Extracts statistics first, standardizes the scales, and then compresses the features using Principal Component Analysis (PCA).
* **Key Test**: Evaluates the effect of dimensionality reduction on generalization.
* **Key Test**: Analyzes the balance between information compression (throwing away redundant sensor data) and overall model accuracy.

---

## Dataset Compatibility & Applicability

Since both pipelines rely on raw continuous EEG waves to perform the initial steps, their compatibility with our three datasets varies:

### 1. Dataset 1 (BEED)
* **Verdict**: ❌ **Not Compatible**
* **Why**: We only have pre-calculated static statistical features. Since there are no raw waves, we cannot apply filtering or PCA compression.

### 2. Dataset 2 (Epileptic Seizure Recognition)
* **Verdict**:  **Compatible**
* **Why**: Each sample contains a raw, continuous single-channel wave (178 time points), allowing us to apply bandpass filtering (Pipeline A) or extract stats and apply PCA (Pipeline B).

### 3. Dataset 3 (EEG Seizure Analysis Dataset)
* **Verdict**:  **Compatible (Highly Recommended)**
* **Why**: Contains raw continuous waves for 23 channels over 250 time points. This is the ideal dataset to test PCA channel compression (Pipeline B) and signal filtering (Pipeline A).
