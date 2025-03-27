import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional

# plt.style.use('seaborn-whitegrid')  # Academic-style plots
plt.style.use('seaborn-v0_8-whitegrid') 
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'figure.autolayout': True,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.format': 'svg'  # Vector format for publications
})

class DualMetricsVisualizer:
    def __init__(self, metrics_dir: str, class_names: Optional[List[str]] = None):
        """
        Args:
            metrics_dir: Directory containing JSON metric files
            class_names: List of class names for confusion matrices
        """
        self.metrics_dir = metrics_dir
        self.class_names = class_names or ['Class 0', 'Class 1', 'Class 2']
        self.metrics = self._load_metrics()

    def _load_metrics(self) -> List[Dict]:
        """Load and sort all metric files."""
        metrics = []
        for fname in sorted(os.listdir(self.metrics_dir)):
            if fname.startswith('metrics_epoch') and fname.endswith('.json'):
                with open(os.path.join(self.metrics_dir, fname)) as f:
                    data = json.load(f)
                    # Convert confusion matrix back to numpy array
                    if 'train' in data and 'confusion_matrix' in data['train']:
                        data['train']['confusion_matrix'] = np.array(data['train']['confusion_matrix'])
                    if 'val' in data and 'confusion_matrix' in data['val']:
                        data['val']['confusion_matrix'] = np.array(data['val']['confusion_matrix'])
                    metrics.append(data)
        return sorted(metrics, key=lambda x: x['epoch'])

    def plot_combined_metrics(self, save_path: Optional[str] = None) -> plt.Figure:
        """Create a 2x2 grid of train vs val metrics.
        
        Returns:
            matplotlib Figure object
        """
        epochs = [m['epoch'] for m in self.metrics]
        
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        
        # Loss Curves
        axs[0,0].plot(epochs, [m['train']['loss'] for m in self.metrics], 
                     label='Train', color='blue', linewidth=2)
        axs[0,0].plot(epochs, [m['val']['loss'] for m in self.metrics], 
                     label='Validation', color='red', linestyle='--', linewidth=2)
        axs[0,0].set_title('Training vs Validation Loss')
        axs[0,0].set_xlabel('Epoch')
        axs[0,0].set_ylabel('Loss')
        axs[0,0].legend()
        
        # Accuracy Curves
        axs[0,1].plot(epochs, [m['train']['accuracy'] for m in self.metrics],
                     color='blue', linewidth=2)
        axs[0,1].plot(epochs, [m['val']['accuracy'] for m in self.metrics],
                     color='red', linestyle='--', linewidth=2)
        axs[0,1].set_title('Accuracy')
        axs[0,1].set_xlabel('Epoch')
        axs[0,1].set_ylabel('Accuracy (%)')
        
        # F1-Score Curves
        axs[1,0].plot(epochs, [m['train']['f1_macro'] for m in self.metrics],
                     color='blue', linewidth=2)
        axs[1,0].plot(epochs, [m['val']['f1_macro'] for m in self.metrics],
                     color='red', linestyle='--', linewidth=2)
        axs[1,0].set_title('F1 Score (Macro)')
        axs[1,0].set_xlabel('Epoch')
        axs[1,0].set_ylabel('F1 Score')
        
        # Highlight Best Epoch
        val_losses = [m['val']['loss'] for m in self.metrics]
        best_epoch = epochs[np.argmin(val_losses)]
        for ax in axs.flat:
            ax.axvline(best_epoch, color='gray', linestyle=':', alpha=0.7)
            ax.annotate(f'Best Epoch: {best_epoch}', 
                       xy=(best_epoch, 0.05), 
                       xycoords=('data', 'axes fraction'),
                       ha='center')
        
        # Confusion Matrix (Best Epoch)
        best_metrics = next(m for m in self.metrics if m['epoch'] == best_epoch)
        cm = best_metrics['val']['confusion_matrix']
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalize
        sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   ax=axs[1,1])
        axs[1,1].set_title(f'Confusion Matrix (Epoch {best_epoch})')
        axs[1,1].set_xlabel('Predicted')
        axs[1,1].set_ylabel('True')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        return fig

    def plot_class_metrics(self, save_path: Optional[str] = None) -> plt.Figure:
        """Plot precision/recall per class for the best epoch."""
        val_losses = [m['val']['loss'] for m in self.metrics]
        best_epoch = self.metrics[np.argmin(val_losses)]
        report = best_epoch['val']['classification_report']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract class metrics (excluding macro/weighted)
        class_data = []
        for class_name in self.class_names:
            if class_name in report:
                class_data.append((
                    class_name,
                    report[class_name]['precision'],
                    report[class_name]['recall'],
                    report[class_name]['f1-score']
                ))
        
        # Convert to numpy array for plotting
        metrics = np.array([x[1:] for x in class_data])
        x = np.arange(len(self.class_names))
        width = 0.25
        
        ax.bar(x - width, metrics[:,0], width, label='Precision')
        ax.bar(x, metrics[:,1], width, label='Recall')
        ax.bar(x + width, metrics[:,2], width, label='F1-Score')
        
        ax.set_title(f'Per-Class Metrics (Epoch {best_epoch["epoch"]})')
        ax.set_xlabel('Class')
        ax.set_ylabel('Score')
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names)
        ax.legend()
        ax.set_ylim(0, 1.1)
        
        # Add value labels
        for i in range(metrics.shape[0]):
            for j, offset in enumerate([-width, 0, width]):
                ax.text(x[i] + offset, metrics[i,j] + 0.02, 
                       f'{metrics[i,j]:.2f}', 
                       ha='center', va='bottom')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        return fig

    def plot_metric_progression(self, metric: str, save_path: Optional[str] = None) -> plt.Figure:
        """Plot a specific metric's progression for both phases.
        
        Args:
            metric: One of ['loss', 'accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
        """
        fig, ax = plt.subplots(figsize=(8, 5))
        
        epochs = [m['epoch'] for m in self.metrics]
        ax.plot(epochs, [m['train'][metric] for m in self.metrics],
                label='Train', color='blue', linewidth=2)
        ax.plot(epochs, [m['val'][metric] for m in self.metrics],
                label='Validation', color='red', linestyle='--', linewidth=2)
        
        # Highlight best epoch
        val_metric = [m['val'][metric] for m in self.metrics]
        best_epoch = epochs[np.argmin(val_metric) if 'loss' in metric else np.argmax(val_metric)]
        ax.axvline(best_epoch, color='gray', linestyle=':')
        ax.annotate(f'Best: Epoch {best_epoch}', 
                   xy=(best_epoch, 0.05), 
                   xycoords=('data', 'axes fraction'),
                   ha='center')
        
        ax.set_title(f'{metric.capitalize()} Progression')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss' if 'loss' in metric else 'Score')
        ax.legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        return fig
    
    def plot_confusion_matrix(self, save_path: Optional[str] = None) -> plt.Figure:
        """Plot confusion matrices for both train and validation for the best epoch."""
        epochs = [m['epoch'] for m in self.metrics]
        val_losses = [m['val']['loss'] for m in self.metrics]
        best_epoch = epochs[np.argmin(val_losses)]
        best_metrics = next(m for m in self.metrics if m['epoch'] == best_epoch)

        fig, axs = plt.subplots(1, 2, figsize=(14, 6))

        # Train Confusion Matrix
        cm_train = best_metrics['train']['confusion_matrix']
        cm_train = cm_train.astype('float') / cm_train.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_train, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names,
                    ax=axs[0])
        axs[0].set_title(f'Train Confusion Matrix (Epoch {best_epoch})')
        axs[0].set_xlabel('Predicted')
        axs[0].set_ylabel('True')

        # Validation Confusion Matrix
        cm_val = best_metrics['val']['confusion_matrix']
        cm_val = cm_val.astype('float') / cm_val.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_val, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names,
                    ax=axs[1])
        axs[1].set_title(f'Validation Confusion Matrix (Epoch {best_epoch})')
        axs[1].set_xlabel('Predicted')
        axs[1].set_ylabel('True')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        return fig

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--metrics_dir', type=str, default='checkpoints/metrics')
    argparser.add_argument('--output_dir', type=str, default='output')
    args = argparser.parse_args()
    viz = DualMetricsVisualizer('checkpoints/metrics', class_names=['Empty', 'Half Full', 'Full'])
    os.makedirs(args.output_dir, exist_ok=True)
    combined_metrics_file = os.path.join(args.output_dir, 'combined_metrics.svg')
    class_metrics_file = os.path.join(args.output_dir, 'class_metrics.svg')
    loss_progression_file = os.path.join(args.output_dir, 'loss_progression.svg')
    accuracy_progression_file = os.path.join(args.output_dir, 'accuracy_progression.svg')
    f1_progression_file = os.path.join(args.output_dir, 'f1_progression.svg')
    confusion_matrices_file = os.path.join(args.output_dir, 'confusion_matrices.svg')
    viz.plot_combined_metrics(combined_metrics_file)
    viz.plot_class_metrics(class_metrics_file)
    viz.plot_metric_progression('loss', loss_progression_file)
    viz.plot_metric_progression('accuracy', accuracy_progression_file)
    viz.plot_metric_progression('f1_macro', f1_progression_file)
    viz.plot_confusion_matrix(confusion_matrices_file)
    print(f"Visualizations saved to {args.output_dir}")