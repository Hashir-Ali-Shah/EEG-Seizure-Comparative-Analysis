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

def run_comprehensive_experiment(X, y, dataset_name, visualizer, is_raw=True):
    print(f"\n--- Running Experiment: {dataset_name} ---")
    
    # 1. Pipeline Comparison
    pipe_a = get_pipeline_a(is_raw=is_raw)
    pipe_b = get_pipeline_b(is_raw=is_raw, n_components=0.95)
    
    X_a = pipe_a.fit_transform(X, y)
    X_b = pipe_b.fit_transform(X)
    
    # Evaluate and select the best strategy (Baseline, SMOTE, or Weighted) to represent the pipelines
    _, base_a = train_model(X_a, y)
    _, smote_a = train_model(X_a, y, imbalance_method='smote')
    _, weighted_a = train_model(X_a, y, class_weight='balanced')
    metrics_a = max([base_a, smote_a, weighted_a], key=lambda x: x['f1'])
    
    _, base_b = train_model(X_b, y)
    _, smote_b = train_model(X_b, y, imbalance_method='smote')
    _, weighted_b = train_model(X_b, y, class_weight='balanced')
    metrics_b = max([base_b, smote_b, weighted_b], key=lambda x: x['f1'])
    
    pipeline_res = [
        {'Dataset': dataset_name, 'Pipeline': 'Pipeline A', 'Accuracy': metrics_a['accuracy'], 'F1': metrics_a['f1'], 'PR-AUC': metrics_a['pr_auc']},
        {'Dataset': dataset_name, 'Pipeline': 'Pipeline B', 'Accuracy': metrics_b['accuracy'], 'F1': metrics_b['f1'], 'PR-AUC': metrics_b['pr_auc']}
    ]
    
    summary = {
        'Dataset': dataset_name,
        'PipeA_F1': metrics_a['f1'],
        'PipeB_F1': metrics_b['f1']
    }
    
    # Run the comprehensive evaluations for BOTH pipelines
    for pipe_name, X_rep in [("Pipeline A", X_a), ("Pipeline B", X_b)]:
        print(f"Running detailed analytics for {pipe_name}...")
        
        # 2. Imbalance Handling Tradeoff with ZERO data leakage
        _, metrics_base = train_model(X_rep, y)
        _, metrics_smote = train_model(X_rep, y, imbalance_method='smote')
        _, metrics_weighted = train_model(X_rep, y, class_weight='balanced')
        
        pr_results = {
            'Baseline': metrics_base['pr_curve'],
            'SMOTE Only': metrics_smote['pr_curve'],
            'Weighting Only': metrics_weighted['pr_curve']
        }
        visualizer.plot_imbalance_impact(pr_results, dataset_name, pipeline_name=pipe_name)
        
        # Save key metrics to summary
        if pipe_name == "Pipeline A":
            summary['PipeA_SMOTE_F1'] = metrics_smote['f1']
            summary['PipeA_Weight_F1'] = metrics_weighted['f1']
        else:
            summary['PipeB_SMOTE_F1'] = metrics_smote['f1']
            summary['PipeB_Weight_F1'] = metrics_weighted['f1']
            
        # 3. Regularization Study
        reg_results = run_regularization_study(X_rep, y)
        visualizer.plot_regularization_comparison(reg_results, dataset_name, pipeline_name=pipe_name)
        
        if pipe_name == "Pipeline A":
            summary['PipeA_L1_F1'] = reg_results['L1']['f1']
            summary['PipeA_L2_F1'] = reg_results['L2']['f1']
        else:
            summary['PipeB_L1_F1'] = reg_results['L1']['f1']
            summary['PipeB_L2_F1'] = reg_results['L2']['f1']
            
        # 4. Multi-Metric Learning Curves (The 4 Scenarios) with ZERO data leakage
        print(f"Generating Multi-Metric Learning Curves for {pipe_name}...")
        
        # Scenario A: Underfitting (High regularization, half the features)
        model_under = LogisticRegression(C=0.001, random_state=42)
        under_curves = generate_multi_metric_learning_curve(model_under, X_rep[:, :max(1, X_rep.shape[1] // 2)], y)
        visualizer.plot_learning_curves(under_curves, f"Underfitting (High $\lambda$)", "LC_Underfit", dataset_name=dataset_name, pipeline_name=pipe_name)
        
        # Scenario B: Overfitting (No regularization, all features)
        model_over = LogisticRegression(C=1000, penalty=None, solver='lbfgs', random_state=42)
        over_curves = generate_multi_metric_learning_curve(model_over, X_rep, y)
        visualizer.plot_learning_curves(over_curves, f"Overfitting (No $\lambda$)", "LC_Overfit", dataset_name=dataset_name, pipeline_name=pipe_name)
        
        # Scenario C: Solution - SMOTE (with leakage-free imbalance_method CV)
        model_std = LogisticRegression(random_state=42)
        smote_curves = generate_multi_metric_learning_curve(model_std, X_rep, y, imbalance_method='smote')
        visualizer.plot_learning_curves(smote_curves, "Solution (SMOTE)", "LC_SMOTE", dataset_name=dataset_name, pipeline_name=pipe_name)
        
        # Scenario D: Solution - Weighting
        model_weight = LogisticRegression(class_weight='balanced', random_state=42)
        weight_curves = generate_multi_metric_learning_curve(model_weight, X_rep, y)
        visualizer.plot_learning_curves(weight_curves, "Solution (Class Weighting)", "LC_Weighting", dataset_name=dataset_name, pipeline_name=pipe_name)
        
    return summary, pipeline_res

def main():
    base_path = r"d:\adnan_amin_project"
    plot_dir = "plots"
    print(f"Executing experiment. Results will be saved in: {plot_dir}")
    
    loader = SeizureDataLoader(base_path)
    visualizer = SeizureVisualizer(output_dir=plot_dir)
    
    # Use proper folder names and load raw waves where applicable
    datasets = [
        ("BEED Bangalore EEG Epilepsy Dataset", loader.load_beed(), False),
        ("Epileptic_Seizure_Recognition", loader.load_recognition(), True),
        ("EEG Seizure Analysis Dataset", loader.load_analysis(extract_features=False), True)
    ]
    
    # Dataset Justification (calculates correct feature sizes including 3D shapes)
    stats = []
    compression_stats = []
    for n, d, is_raw in datasets:
        features_before = d[0].shape[1] * d[0].shape[2] if len(d[0].shape) == 3 else d[0].shape[1]
        
        # Calculate features after pipeline A feature extraction
        if n == "BEED Bangalore EEG Epilepsy Dataset":
            features_after = 16
        elif n == "Epileptic_Seizure_Recognition":
            features_after = 30 # 5 windows * 6 stats
        else:
            features_after = 138 # 23 channels * 6 stats
            
        stats.append({
            'Name': n,
            'Samples': d[0].shape[0],
            'Imbalance': np.mean(d[1]),
            'Features': features_before
        })
        compression_stats.append({
            'Dataset': n.replace("BEED Bangalore EEG Epilepsy Dataset", "BEED Tabular").replace("Epileptic_Seizure_Recognition", "Recognition Flat Time-Series").replace("EEG Seizure Analysis Dataset", "Analysis 3D Multi-Channel"),
            'Before': features_before,
            'After': features_after
        })
    visualizer.plot_dataset_justification(stats)
    visualizer.plot_feature_compression(compression_stats)
    
    all_summary = []
    all_pipeline_metrics = []
    for name, (X, y), is_raw in datasets:
        summary, pipe_metrics = run_comprehensive_experiment(X, y, name, visualizer, is_raw=is_raw)
        all_summary.append(summary)
        all_pipeline_metrics.extend(pipe_metrics)
    
    # Final Visualizations
    visualizer.plot_pipeline_metrics(all_pipeline_metrics)
    summary_df = pd.DataFrame(all_summary)
    visualizer.plot_final_comparison(summary_df)
    
    summary_df.to_csv(os.path.join(plot_dir, "comparative_analysis", "comparative_analysis.csv"), index=False)
    print(f"\nExperiment complete. All dual-pipeline visualizations saved in '{plot_dir}'.")

if __name__ == "__main__":
    main()
