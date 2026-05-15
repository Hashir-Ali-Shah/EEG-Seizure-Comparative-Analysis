from data_loader import SeizureDataLoader
from pipelines import get_pipeline_a, get_pipeline_b
from imbalance_handler import handle_imbalance
from model_engine import train_model, run_regularization_study, generate_multi_metric_learning_curve
from visualizer import SeizureVisualizer
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import os

def get_next_plot_dir(base_name="plots"):
    """Finds the next available directory name by incrementing a counter."""
    if not os.path.exists(base_name):
        return base_name
    counter = 1
    while os.path.exists(f"{base_name}_{counter}"):
        counter += 1
    return f"{base_name}_{counter}"

def run_comprehensive_experiment(X, y, dataset_name, visualizer):
    print(f"\n--- Running Experiment: {dataset_name} ---")
    
    # 1. Pipeline Comparison
    pipe_a = get_pipeline_a(n_features=min(20, X.shape[1]))
    pipe_b = get_pipeline_b(n_components=0.95)
    X_a = pipe_a.fit_transform(X, y)
    X_b = pipe_b.fit_transform(X)
    _, metrics_a = train_model(X_a, y)
    _, metrics_b = train_model(X_b, y)
    
    pipeline_res = [
        {'Dataset': dataset_name, 'Pipeline': 'Pipeline A', 'Accuracy': metrics_a['accuracy'], 'F1': metrics_a['f1'], 'PR-AUC': metrics_a['pr_auc']},
        {'Dataset': dataset_name, 'Pipeline': 'Pipeline B', 'Accuracy': metrics_b['accuracy'], 'F1': metrics_b['f1'], 'PR-AUC': metrics_b['pr_auc']}
    ]
    
    # 2. Imbalance Handling Tradeoff
    _, metrics_base = train_model(X_a, y)
    X_smote, y_smote = handle_imbalance(X_a, y, method='smote')
    _, metrics_smote = train_model(X_smote, y_smote)
    _, metrics_weighted = train_model(X_a, y, class_weight='balanced')
    
    pr_results = {
        'Baseline': metrics_base['pr_curve'],
        'SMOTE Only': metrics_smote['pr_curve'],
        'Weighting Only': metrics_weighted['pr_curve']
    }
    visualizer.plot_imbalance_impact(pr_results, dataset_name)
    
    # 3. Regularization Study
    reg_results = run_regularization_study(X_a, y)
    visualizer.plot_regularization_comparison(reg_results, dataset_name)
    
    # 4. Multi-Metric Learning Curves (The 4 Scenarios)
    print("Generating Multi-Metric Learning Curves...")
    
    # Scenario A: Underfitting (High regularization, half the features)
    model_under = LogisticRegression(C=0.001, random_state=42)
    under_curves = generate_multi_metric_learning_curve(model_under, X[:, :max(1, X.shape[1] // 2)], y)
    visualizer.plot_learning_curves(under_curves, f"Underfitting (High $\lambda$)", f"{dataset_name}_LC_Underfit")
    
    # Scenario B: Overfitting (No regularization, all features)
    model_over = LogisticRegression(C=1000, penalty=None, solver='lbfgs', random_state=42)
    over_curves = generate_multi_metric_learning_curve(model_over, X, y)
    visualizer.plot_learning_curves(over_curves, f"Overfitting (No $\lambda$)", f"{dataset_name}_LC_Overfit")
    
    # Scenario C: Solution - SMOTE
    model_std = LogisticRegression(random_state=42)
    smote_curves = generate_multi_metric_learning_curve(model_std, X_smote, y_smote)
    visualizer.plot_learning_curves(smote_curves, "Solution (SMOTE)", f"{dataset_name}_LC_SMOTE")
    
    # Scenario D: Solution - Weighting
    model_weight = LogisticRegression(class_weight='balanced', random_state=42)
    weight_curves = generate_multi_metric_learning_curve(model_weight, X, y)
    visualizer.plot_learning_curves(weight_curves, "Solution (Class Weighting)", f"{dataset_name}_LC_Weighting")
    
    return {
        'Dataset': dataset_name,
        'PipeA_F1': metrics_a['f1'],
        'PipeB_F1': metrics_b['f1'],
        'SMOTE_F1': metrics_smote['f1'],
        'Weight_F1': metrics_weighted['f1'],
        'L1_F1': reg_results['L1']['f1'],
        'L2_F1': reg_results['L2']['f1']
    }, pipeline_res

def main():
    base_path = r"d:\adnan_amin_project"
    plot_dir = get_next_plot_dir()
    print(f"Executing experiment. Results will be saved in: {plot_dir}")
    
    loader = SeizureDataLoader(base_path)
    visualizer = SeizureVisualizer(output_dir=plot_dir)
    
    # Use proper folder names for justification
    datasets = [
        ("BEED Bangalore EEG Epilepsy Dataset", loader.load_beed()),
        ("Epileptic_Seizure_Recognition", loader.load_recognition()),
        ("EEG Seizure Analysis Dataset", loader.load_analysis(extract_features=True))
    ]
    
    # Dataset Justification
    stats = [{'Name': n, 'Samples': d[0].shape[0], 'Imbalance': np.mean(d[1]), 'Features': d[0].shape[1]} for n, d in datasets]
    visualizer.plot_dataset_justification(stats)
    
    all_summary = []
    all_pipeline_metrics = []
    for name, (X, y) in datasets:
        summary, pipe_metrics = run_comprehensive_experiment(X, y, name, visualizer)
        all_summary.append(summary)
        all_pipeline_metrics.extend(pipe_metrics)
    
    # Final Visualizations
    visualizer.plot_pipeline_metrics(all_pipeline_metrics)
    summary_df = pd.DataFrame(all_summary)
    visualizer.plot_final_comparison(summary_df)
    
    summary_df.to_csv(os.path.join(plot_dir, "comparative_analysis.csv"), index=False)
    print(f"\nExperiment complete. All 21+ visualizations saved in '{plot_dir}'.")

if __name__ == "__main__":
    main()
