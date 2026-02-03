"""
Evaluation metrics and performance analysis for URL phishing detection models.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, 
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    """
    Calculate comprehensive metrics for binary classification.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels (binary)
        y_pred_proba: Predicted probabilities (optional, for AUC)
    
    Returns:
        Dictionary containing all metrics
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0
    
    metrics = {
        'confusion_matrix': cm,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'sensitivity': recall_score(y_true, y_pred, zero_division=0),
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0,
        'fnr': fn / (fn + tp) if (fn + tp) > 0 else 0,
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn)
    }
    
    # Add AUC if probabilities provided
    if y_pred_proba is not None:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
        except:
            metrics['auc'] = None
    
    return metrics


def print_metrics(metrics, title="Metrics"):
    """
    Print metrics in a formatted table.
    
    Args:
        metrics: Dictionary of metrics
        title: Title for the metrics display
    """
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")
    
    # Classification metrics
    print(f"\n{'Metric':<25} {'Value':<15}")
    print(f"{'-'*40}")
    
    metric_names = [
        ('Accuracy', 'accuracy'),
        ('Precision', 'precision'),
        ('Recall (Sensitivity)', 'recall'),
        ('F1-Score', 'f1_score'),
        ('Specificity', 'specificity'),
        ('False Positive Rate', 'fpr'),
        ('False Negative Rate', 'fnr'),
    ]
    
    for display_name, key in metric_names:
        if key in metrics:
            value = metrics[key]
            if isinstance(value, dict):
                value = value.get('mean', 0)
            print(f"{display_name:<25} {value:.4f}")
    
    if 'auc' in metrics and metrics['auc'] is not None:
        print(f"{'AUC-ROC':<25} {metrics['auc']:.4f}")
    
    # Confusion matrix
    if 'confusion_matrix' in metrics:
        cm = metrics['confusion_matrix']
        if cm.shape == (2, 2):
            print(f"\n{'Confusion Matrix':^40}")
            print(f"{'-'*40}")
            print(f"{'':20} {'Predicted':>20}")
            print(f"{'':20} {'Benign':>10} {'Malicious':>10}")
            print(f"{'Actual':20} {'Benign':>10} {cm[0,0]:>10} {cm[0,1]:>10}")
            print(f"{'':20} {'Malicious':>10} {cm[1,0]:>10} {cm[1,1]:>10}")
    
    print(f"{'='*60}\n")


def plot_confusion_matrix(cm, title="Confusion Matrix", figsize=(8, 6)):
    """
    Plot confusion matrix as heatmap.
    
    Args:
        cm: Confusion matrix
        title: Plot title
        figsize: Figure size
    
    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Benign', 'Malicious'],
        yticklabels=['Benign', 'Malicious'],
        ax=ax
    )
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_roc_curve(y_true, y_pred_proba, title="ROC Curve", figsize=(8, 6)):
    """
    Plot ROC curve.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        title: Plot title
        figsize: Figure size
    
    Returns:
        matplotlib figure
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(fpr, tpr, linewidth=2, label=f'AUC = {auc:.4f}')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig


def compare_models(model_results, metric='accuracy'):
    """
    Compare multiple models based on a specific metric.
    
    Args:
        model_results: Dictionary of {model_name: metrics_dict}
        metric: Metric to compare ('accuracy', 'f1_score', etc.)
    
    Returns:
        DataFrame with comparison results
    """
    comparison_data = []
    
    for model_name, metrics in model_results.items():
        if metric in metrics:
            value = metrics[metric]
            if isinstance(value, dict):
                value = value.get('mean', 0)
            comparison_data.append({
                'Model': model_name,
                'Metric': metric,
                'Value': value
            })
    
    df = pd.DataFrame(comparison_data)
    df = df.sort_values('Value', ascending=False).reset_index(drop=True)
    
    return df


def plot_model_comparison(model_results, metrics=['accuracy', 'precision', 'recall', 'f1_score'],
                         figsize=(12, 6)):
    """
    Plot bar chart comparing models across multiple metrics.
    
    Args:
        model_results: Dictionary of {model_name: metrics_dict}
        metrics: List of metrics to compare
        figsize: Figure size
    
    Returns:
        matplotlib figure
    """
    data = []
    for model_name, model_metrics in model_results.items():
        for metric in metrics:
            if metric in model_metrics:
                value = model_metrics[metric]
                if isinstance(value, dict):
                    value = value.get('mean', 0)
                data.append({
                    'Model': model_name,
                    'Metric': metric.replace('_', ' ').title(),
                    'Value': value
                })
    
    df = pd.DataFrame(data)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create grouped bar chart
    models = df['Model'].unique()
    metric_names = df['Metric'].unique()
    x = np.arange(len(models))
    width = 0.8 / len(metric_names)
    
    for i, metric_name in enumerate(metric_names):
        metric_data = df[df['Metric'] == metric_name]
        values = [metric_data[metric_data['Model'] == m]['Value'].values[0] 
                 if len(metric_data[metric_data['Model'] == m]) > 0 else 0 
                 for m in models]
        ax.bar(x + i * width, values, width, label=metric_name)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * (len(metric_names) - 1) / 2)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    return fig


def create_classification_report(y_true, y_pred, target_names=None):
    """
    Generate detailed classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        target_names: Names for classes (default: ['Benign', 'Malicious'])
    
    Returns:
        Classification report as string
    """
    if target_names is None:
        target_names = ['Benign', 'Malicious']
    
    report = classification_report(
        y_true, y_pred,
        target_names=target_names,
        digits=4
    )
    
    return report


def calculate_cv_statistics(cv_results):
    """
    Calculate statistics across cross-validation folds.
    
    Args:
        cv_results: List of metrics dictionaries from each fold
    
    Returns:
        Dictionary with mean, std, min, max for each metric
    """
    if not cv_results:
        return {}
    
    stats = {}
    metric_keys = ['accuracy', 'precision', 'recall', 'f1_score', 
                   'sensitivity', 'specificity', 'fpr', 'fnr']
    
    for key in metric_keys:
        values = [result[key] for result in cv_results if key in result]
        if values:
            stats[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'values': values
            }
    
    return stats


def evaluate_model(y_true, y_pred, y_pred_proba=None, model_name="Model"):
    """
    Complete evaluation of a model with metrics and visualization.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (optional)
        model_name: Name of the model for display
    
    Returns:
        Dictionary of metrics and figures
    """
    metrics = calculate_metrics(y_true, y_pred, y_pred_proba)
    
    print_metrics(metrics, title=f"{model_name} Performance")
    
    results = {
        'metrics': metrics,
        'figures': {}
    }
    
    # Confusion matrix plot
    results['figures']['confusion_matrix'] = plot_confusion_matrix(
        metrics['confusion_matrix'],
        title=f"{model_name} - Confusion Matrix"
    )
    
    # ROC curve if probabilities available
    if y_pred_proba is not None:
        try:
            results['figures']['roc_curve'] = plot_roc_curve(
                y_true, y_pred_proba,
                title=f"{model_name} - ROC Curve"
            )
        except:
            pass
    
    return results
