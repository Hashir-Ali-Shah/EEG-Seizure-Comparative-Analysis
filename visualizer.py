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

    def _get_save_path(self, filename, dataset_name=None):
        """Helper to resolve clean save paths inside dataset subfolders or comparative_analysis folder."""
        if dataset_name:
            dataset_dir = os.path.join(self.output_dir, dataset_name)
        else:
            dataset_dir = os.path.join(self.output_dir, "comparative_analysis")
            
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
        return os.path.join(dataset_dir, filename)

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
        plt.savefig(self._get_save_path("dataset_justification.png"))
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
        plt.savefig(self._get_save_path("pipeline_metrics_comparison.png"))
        plt.close()

    def plot_learning_curves(self, curve_data, title, filename, dataset_name=None, pipeline_name=None):
        """
        Plots learning curves for multiple metrics (F1, Precision, Recall).
        curve_data: dict with metrics as keys, each containing (train_sizes, train_scores, val_scores)
        """
        metrics = list(curve_data.keys())
        fig, axes = plt.subplots(1, len(metrics), figsize=(7 * len(metrics), 5))
        if len(metrics) == 1: axes = [axes]

        plot_title = f"{title} ({pipeline_name})" if pipeline_name else title
        save_name = f"{filename}_{pipeline_name.replace(' ', '_')}" if pipeline_name else filename

        for i, metric in enumerate(metrics):
            train_sizes, train_scores, val_scores = curve_data[metric]
            axes[i].plot(train_sizes, train_scores, 'o-', label=f"Training {metric}")
            axes[i].plot(train_sizes, val_scores, 'o-', label=f"Validation {metric}")
            axes[i].set_title(f"{plot_title} - {metric}")
            axes[i].set_xlabel("Training Examples")
            axes[i].set_ylabel(metric)
            axes[i].legend()

        plt.tight_layout()
        plt.savefig(self._get_save_path(f"{save_name}.png", dataset_name))
        plt.close()

    def plot_regularization_comparison(self, results, dataset_name, pipeline_name=None):
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
        
        plot_title = f"Regularization Sparsity vs Performance ({dataset_name} - {pipeline_name})" if pipeline_name else f"Regularization Sparsity vs Performance ({dataset_name})"
        save_name = f"regularization_sparsity_{pipeline_name.replace(' ', '_')}.png" if pipeline_name else "regularization_sparsity.png"
        
        plt.title(plot_title)
        plt.tight_layout()
        plt.savefig(self._get_save_path(save_name, dataset_name))
        plt.close()

    def plot_imbalance_impact(self, pr_results, dataset_name, pipeline_name=None):
        """
        Plots PR curves for different imbalance handling techniques.
        """
        plt.figure(figsize=(10, 7))
        for method, (prec, rec) in pr_results.items():
            linestyle = '--' if method == 'Baseline' else '-'
            plt.plot(rec, prec, label=method, linestyle=linestyle)
            
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plot_title = f'Precision-Recall Tradeoff ({dataset_name} - {pipeline_name})' if pipeline_name else f'Precision-Recall Tradeoff ({dataset_name})'
        save_name = f"imbalance_pr_tradeoff_{pipeline_name.replace(' ', '_')}.png" if pipeline_name else "imbalance_pr_tradeoff.png"
        
        plt.title(plot_title)
        plt.legend()
        plt.grid(True)
        plt.savefig(self._get_save_path(save_name, dataset_name))
        plt.close()

    def plot_final_comparison(self, summary_df):
        """Final summary heatmap of F1 scores across datasets and experiments."""
        plt.figure(figsize=(12, 6))
        plot_df = summary_df.set_index('Dataset')
        sns.heatmap(plot_df, annot=True, cmap='YlGnBu', fmt=".3f")
        plt.title('Final Comparative Analysis (F1 Scores)')
        plt.tight_layout()
        plt.savefig(self._get_save_path("final_comparative_analysis.png"))
        plt.close()

    def plot_feature_compression(self, compression_stats):
        """
        Plots a comparative grouped bar chart showing feature counts before and after processing.
        compression_stats: List of dicts with 'Dataset', 'Before', 'After'
        """
        df = pd.DataFrame(compression_stats)
        df_melted = df.melt(id_vars='Dataset', value_vars=['Before', 'After'], 
                             var_name='Stage', value_name='Features')
        
        plt.figure(figsize=(12, 7))
        # Sleek color scheme (muted but distinctive)
        ax = sns.barplot(data=df_melted, x='Dataset', y='Features', hue='Stage', palette='Set2')
        
        # Using logarithmic scale so all feature sizes are clearly visible
        ax.set_yscale('log')
        
        # Display the exact counts on top of each bar
        for p in ax.patches:
            val = p.get_height()
            if val > 0:
                ax.annotate(f"{int(val)}", 
                            (p.get_x() + p.get_width() / 2., val), 
                            ha='center', va='bottom', 
                            xytext=(0, 5), 
                            textcoords='offset points',
                            fontsize=11, weight='bold', color='#2F4F4F')
                
        plt.title('Feature Dimensionality: Before vs After Preprocessing', fontsize=15, weight='bold')
        plt.ylabel('Feature Count (Logarithmic Scale)', fontsize=12)
        plt.xlabel('Dataset', fontsize=12)
        plt.tick_params(axis='x', labelsize=11)
        plt.legend(title='Preprocessing Stage', title_fontsize='11', fontsize='10')
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(self._get_save_path("feature_compression_before_after.png"))
        plt.close()
