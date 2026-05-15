import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

class SeizureVisualizer:
    def __init__(self, output_dir="plots"):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        sns.set_theme(style="whitegrid")

    def plot_dataset_justification(self, dataset_stats):
        """
        Plots justification for datasets: Size, Imbalance, Feature count.
        dataset_stats: List of dicts with 'Name', 'Samples', 'Imbalance', 'Features'
        """
        df = pd.DataFrame(dataset_stats)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        sns.barplot(data=df, x='Name', y='Samples', ax=axes[0], hue='Name', palette='viridis', legend=False)
        axes[0].set_title('Dataset Size (Samples)')
        axes[0].tick_params(axis='x', rotation=45)
        
        sns.barplot(data=df, x='Name', y='Imbalance', ax=axes[1], hue='Name', palette='magma', legend=False)
        axes[1].set_title('Class Imbalance (Ratio Seizure:Total)')
        axes[1].tick_params(axis='x', rotation=45)
        
        sns.barplot(data=df, x='Name', y='Features', ax=axes[2], hue='Name', palette='rocket', legend=False)
        axes[2].set_title('Feature Count (Characteristics)')
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "dataset_justification.png"))
        plt.close()

    def plot_pipeline_metrics(self, pipeline_results):
        """
        Plots Accuracy, F1, and PR-AUC for each pipeline across datasets.
        """
        df = pd.DataFrame(pipeline_results)
        metrics = ['Accuracy', 'F1', 'PR-AUC']
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        for i, metric in enumerate(metrics):
            sns.barplot(data=df, x='Dataset', y=metric, hue='Pipeline', ax=axes[i])
            axes[i].set_title(f'{metric} Comparison')
            axes[i].set_ylim(0, 1.1)
            axes[i].tick_params(axis='x', rotation=30)
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "pipeline_metrics_comparison.png"))
        plt.close()

    def plot_learning_curves(self, curve_data, title, filename):
        """
        Plots learning curves for multiple metrics (F1, Precision, Recall).
        curve_data: dict with metrics as keys, each containing (train_sizes, train_scores, val_scores)
        """
        metrics = list(curve_data.keys())
        fig, axes = plt.subplots(1, len(metrics), figsize=(7 * len(metrics), 5))
        if len(metrics) == 1: axes = [axes]

        for i, metric in enumerate(metrics):
            train_sizes, train_scores, val_scores = curve_data[metric]
            axes[i].plot(train_sizes, train_scores, 'o-', label=f"Training {metric}")
            axes[i].plot(train_sizes, val_scores, 'o-', label=f"Validation {metric}")
            axes[i].set_title(f"{title} - {metric}")
            axes[i].set_xlabel("Training Examples")
            axes[i].set_ylabel(metric)
            axes[i].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{filename}.png"))
        plt.close()

    def plot_regularization_comparison(self, results, dataset_name):
        """Plots a comparison of different regularization techniques."""
        labels = list(results.keys())
        f1_scores = [results[l]['f1'] for l in labels]
        sparsity = [results[l]['sparsity'] for l in labels]
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        color = 'tab:blue'
        ax1.set_xlabel('Regularization Type')
        ax1.set_ylabel('F1 Score', color=color)
        ax1.bar(labels, f1_scores, color=color, alpha=0.6, label='F1 Score')
        ax1.tick_params(axis='y', labelcolor=color)
        
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Sparsity (%)', color=color)
        ax2.plot(labels, sparsity, color=color, marker='D', label='Sparsity')
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title(f"Regularization Sparsity vs Performance ({dataset_name})")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{dataset_name}_regularization_sparsity.png"))
        plt.close()

    def plot_imbalance_impact(self, pr_results, dataset_name):
        """
        Plots PR curves for different imbalance handling techniques.
        """
        plt.figure(figsize=(10, 7))
        for method, (prec, rec) in pr_results.items():
            # Use dashed line for Baseline to see SMOTE overlaps if any
            linestyle = '--' if method == 'Baseline' else '-'
            plt.plot(rec, prec, label=method, linestyle=linestyle)
            
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Tradeoff ({dataset_name})')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, f"{dataset_name}_imbalance_pr_tradeoff.png"))
        plt.close()

    def plot_final_comparison(self, summary_df):
        """Final summary heatmap of F1 scores across datasets and experiments."""
        plt.figure(figsize=(12, 6))
        plot_df = summary_df.set_index('Dataset')
        sns.heatmap(plot_df, annot=True, cmap='YlGnBu', fmt=".3f")
        plt.title('Final Comparative Analysis (F1 Scores)')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "final_comparative_analysis.png"))
        plt.close()
