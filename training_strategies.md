# Training & Class Imbalance Strategies

This document details the three distinct training strategies used to handle class imbalance in our seizure prediction models.

---

## Strategy 1: Baseline Training (Without Handling Imbalance)

### 1. Description
The Logistic Regression model is trained directly on the raw, unbalanced datasets with no adjustments.

### 2. Core Purpose & Tests
* **Establish a Baseline**: Provides a clean performance reference point to see how the model behaves out of the box.
* **Demonstrate the Accuracy Trap**: Shows how accuracy can look high (e.g. 85%) while the F1-score and Precision-Recall metrics remain extremely low due to the model ignoring the rare seizure class.

---

## Strategy 2: Cost-Sensitive Training (Class Weighting)

### 1. Description
We train the model using Scikit-Learn's `class_weight='balanced'` parameter. This tells the Logistic Regression solver to mathematically penalize mistakes on the rare seizure class much more heavily than mistakes on the common non-seizure class.

### 2. Core Purpose & Tests
* **No Synthetic Data**: Solves the imbalance strictly inside the loss function without creating artificial data points.
* **Generalization**: Evaluates if weighting minority class errors is sufficient to improve seizure recall without generating too many false alarms (Precision drop).

---

## Strategy 3: Algorithmic Balancing (SMOTE Oversampling)

### 1. Description
Prior to training, we apply the **Synthetic Minority Over-sampling Technique (SMOTE)** to create new, synthetic patient samples of the rare seizure class until the dataset has a perfect 50/50 balance.

### 2. Core Purpose & Tests
* **Direct Data Balancing**: Balances the class distribution directly in the data space before the model even sees it.
* **High-Dimensional Support**: Evaluates if adding synthetic variations of seizure waves helps the model build a more robust decision boundary compared to simply increasing minority weights.
