from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, auc, precision_score, recall_score
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.impute import SimpleImputer
import numpy as np

def calculate_pr_auc(y_true, y_prob):
    """Calculates Area Under Precision-Recall Curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    return auc(recall, precision)

def train_model(X, y, penalty='l2', C=1.0, solver='lbfgs', class_weight=None):
    """Trains a Logistic Regression model and returns detailed metrics."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Pre-impute just in case
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    model = LogisticRegression(penalty=penalty, C=C, solver=solver, max_iter=1000, random_state=42, class_weight=class_weight)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'pr_auc': calculate_pr_auc(y_test, y_prob),
        'pr_curve': (precision, recall)
    }
    
    return model, metrics

def generate_multi_metric_learning_curve(model, X, y):
    """Generates learning curve data for F1, Precision, and Recall."""
    # Pre-impute
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    metrics = ['f1', 'precision', 'recall']
    results = {}
    
    for metric in metrics:
        train_sizes, train_scores, val_scores = learning_curve(
            model, X_imputed, y, cv=5, scoring=metric, train_sizes=np.linspace(0.1, 1.0, 5)
        )
        results[metric] = (train_sizes, np.mean(train_scores, axis=1), np.mean(val_scores, axis=1))
    
    return results

def run_regularization_study(X, y):
    """Compares L1, L2, and Elastic Net regularization."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Impute
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    results = {}
    
    # L1
    model_l1 = LogisticRegression(penalty='l1', C=0.5, solver='liblinear', random_state=42)
    model_l1.fit(X_train, y_train)
    sparsity = np.mean(model_l1.coef_ == 0) * 100
    results['L1'] = {'f1': f1_score(y_test, model_l1.predict(X_test), zero_division=0), 'sparsity': sparsity}
    
    # L2
    model_l2 = LogisticRegression(penalty='l2', C=0.5, solver='lbfgs', random_state=42)
    model_l2.fit(X_train, y_train)
    results['L2'] = {'f1': f1_score(y_test, model_l2.predict(X_test), zero_division=0), 'sparsity': 0.0}
    
    # Elastic Net
    model_en = LogisticRegression(penalty='elasticnet', l1_ratio=0.5, C=0.5, solver='saga', max_iter=2000, random_state=42)
    model_en.fit(X_train, y_train)
    results['ElasticNet'] = {'f1': f1_score(y_test, model_en.predict(X_test), zero_division=0), 'sparsity': np.mean(model_en.coef_ == 0) * 100}
    
    return results
