# EEG-Seizure-Comparative-Analysis

## Overview
This project presents a rigorous comparative analysis of machine learning strategies for **Epileptic Seizure Prediction**. By evaluating Logistic Regression models across three distinct EEG datasets, the study investigates how preprocessing order, regularization techniques, and class imbalance handling interact to affect model generalization and stability.

## Key Research Objectives
- **Preprocessing Impact**: Evaluating the performance difference between normalization-led (Pipeline A) and dimensionality-reduction-led (Pipeline B) strategies.
- **Generalization Analysis**: Diagnosing model fit through multi-metric learning curves comparing Underfitting (High $\lambda$, half-features) and Overfitting (No $\lambda$, all features) scenarios.
- **Regularization Sparsity**: Analyzing the feature-selection effects of L1 (Lasso) vs. L2 (Ridge) and Elastic Net penalties.
- **Class Imbalance Mitigation**: Quantifying the Precision-Recall tradeoff using SMOTE and Class Weighting techniques.

---

## Datasets Analyzed
1.  **BEED Bangalore EEG Epilepsy Dataset**: 8,000 samples, high-quality information density, low feature-to-sample ratio.
2.  **Epileptic Seizure Recognition**: 11,500 samples, high-dimensional time-series data (178 features).
3.  **EEG Seizure Analysis Dataset**: 8,282 samples of 23-channel 3D signal data (processed via statistical feature extraction).

---

## Implementation Details

### 1. Preprocessing Pipelines
- **Pipeline A**: `Imputer` → `StandardScaler` → `OutlierClipper` → `SelectKBest`.
- **Pipeline B**: `Imputer` → `MinMaxScaler` → `PCA`.

### 2. Regularization Study
A systematic comparison of:
- **L1 (Lasso)**: For inducing feature sparsity and selection.
- **L2 (Ridge)**: For weight stabilization.
- **Elastic Net**: For a balanced combination of sparsity and stability.

### 3. Handling Class Imbalance
- **SMOTE**: Synthetic Minority Over-sampling Technique.
- **Class Weighting**: Cost-sensitive learning using balanced weights.
- **Precision-Recall Tradeoff Analysis**: Automated PR-Curve generation for all handling strategies.

---

## Key Results & Visualizations

### 1. Dataset Characteristics
The following chart justifies the dataset selection based on sample size, imbalance ratio, and feature complexity.
![Dataset Justification](plots_3/dataset_justification.png)

### 2. Preprocessing Pipeline Performance
Comparison of Accuracy, F1-Score, and PR-AUC across Pipeline A and Pipeline B.
![Pipeline Comparison](plots_3/pipeline_metrics_comparison.png)

### 3. Generalization & Learning Trajectories (Recognition Dataset)
Comparison of High Bias (Underfitting) vs. High Variance (Overfitting) as training data scales.
![Learning Curves](plots_3/Epileptic_Seizure_Recognition_LC_Overfit.png)

### 4. Precision-Recall Tradeoff (Analysis Dataset)
Visualizing the impact of SMOTE and Class Weighting on model reliability.
![PR Tradeoff](plots_3/EEG%20Seizure%20Analysis%20Dataset_imbalance_pr_tradeoff.png)

### 5. Final Comparative Summary
Heatmap of F1-scores across all datasets and experimental conditions.
![Final Heatmap](plots_3/final_comparative_analysis.png)

---

## Key Insights
- **Imbalance Handling**: SMOTE consistently provided the highest F1-score gains in high-dimensional datasets (Recognition and Analysis), boosting performance by over 500% in some cases.
- **Sparsity vs. Stability**: L1 regularization successfully reduced feature counts without significant performance degradation, identifying redundant EEG features.
- **Learning Trajectory**: Observations across all datasets showed identical performance "shapes" across different fit scenarios, indicating that the intrinsic complexity of the EEG data is the primary performance constraint.

---

## Project Structure
- `data_loader.py`: Unified loading logic for CSV and NPZ formats.
- `pipelines.py`: Modular Scikit-Learn preprocessing pipelines.
- `model_engine.py`: Core logic for training, regularization studies, and fit simulations.
- `visualizer.py`: Custom Matplotlib/Seaborn visualization suite.
- `main.py`: Orchestration script for the entire experimental pipeline.

---

## How to Run
1. Ensure dependencies are installed: `pip install numpy pandas scikit-learn matplotlib seaborn imbalanced-learn`
2. Run the full experiment suite:
   ```bash
   python main.py
   ```
3. Results and over 20+ visualizations will be automatically saved in a new `plots_X` folder.
