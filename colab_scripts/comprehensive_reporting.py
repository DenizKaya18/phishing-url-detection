import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
import os
warnings.filterwarnings('ignore')

class ComprehensiveReporting:
    """Comprehensive visualization and reporting for ensemble models"""
    
    def __init__(self, classifier, save_dir="reports/"):
        self.classifier = classifier
        self.save_dir = save_dir
        self.cv_scores = classifier.cv_scores
        self.model_info = classifier.model_info
        self.cv_metrics = classifier.cv_metrics
        self.final_metrics = classifier.final_metrics
        self.final_confusion_matrix = classifier.final_confusion_matrix
        self.avg_confusion_matrix = classifier.avg_confusion_matrix
        
        # Create save directory
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
    def plot_cv_accuracies_boxplot(self, figsize=(14, 8)):
        """
        Plot accuracies of different architectures across 10-fold CV with box plots
        Shows average in parenthesis
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        model_types = list(self.cv_scores['individual'].keys())
        model_types.append('Ensemble')
        
        # Prepare data
        box_data = []
        labels = []
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
        
        for i, model_type in enumerate(model_types):
            if model_type == 'Ensemble':
                scores = self.cv_scores['ensemble']
            else:
                scores = self.cv_scores['individual'][model_type]
            
            box_data.append(scores)
            avg_score = np.mean(scores)
            labels.append(f"{model_type}\n({avg_score:.4f})")
        
        # Create box plot
        bp = ax.boxplot(box_data, labels=labels, patch_artist=True,
                        widths=0.6, showmeans=True,
                        meanprops=dict(marker='D', markerfacecolor='red', 
                                     markeredgecolor='red', markersize=8))
        
        # Color the boxes
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Styling
        ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_title('Accuracies of Different Deep Learning Model Architectures\nAcross 10-Fold Cross Validation',
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_ylim([0.95, 1.005])
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Add value labels on points
        for i, (data, color) in enumerate(zip(box_data, colors)):
            y = data
            x = np.random.normal(i+1, 0.04, size=len(y))
            ax.scatter(x, y, alpha=0.4, s=50, color=color)
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}cv_accuracies_boxplot.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ CV Accuracies Box Plot saved")
    
    def plot_cv_accuracies_line(self, figsize=(14, 8)):
        """
        Line plot showing accuracy variation across folds
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        model_types = list(self.cv_scores['individual'].keys())
        folds = np.arange(1, len(self.cv_scores['ensemble']) + 1)
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
        markers = ['o', 's', '^', 'D']
        
        # Plot individual models
        for model_type, color, marker in zip(model_types, colors, markers):
            scores = self.cv_scores['individual'][model_type]
            mean_score = np.mean(scores)
            ax.plot(folds, scores, marker=marker, label=f'{model_type} ({mean_score:.4f})',
                   color=color, linewidth=2.5, markersize=8, alpha=0.8)
        
        # Plot ensemble
        ensemble_scores = self.cv_scores['ensemble']
        ensemble_mean = np.mean(ensemble_scores)
        ax.plot(folds, ensemble_scores, marker='*', label=f'Ensemble ({ensemble_mean:.4f})',
               color='#98D8C8', linewidth=3, markersize=15, alpha=0.9)
        
        # Styling
        ax.set_xlabel('Fold Number', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_title('Accuracies of Different Deep Learning Model Architectures\nAcross 10-Fold Cross Validation',
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(folds)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='lower right', fontsize=11, framealpha=0.95)
        ax.set_ylim([0.95, 1.005])
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}cv_accuracies_line.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ CV Accuracies Line Plot saved")
    
    def create_training_efficiency_table(self):
        """
        Create comprehensive training efficiency table
        """
        data = []
        
        model_types = list(self.cv_scores['individual'].keys())
        
        for model_type in model_types:
            cv_mean = np.mean(self.cv_scores['individual'][model_type])
            cv_std = np.std(self.cv_scores['individual'][model_type])
            
            # Find model info
            model_idx = None
            for idx, info in enumerate(self.model_info):
                if info['type'] == model_type:
                    model_idx = idx
                    break
            
            if model_idx is not None:
                info = self.model_info[model_idx]
                val_acc = info.get('val_accuracy', 0)
                test_acc = info.get('test_accuracy', 0)
                epochs = info.get('epochs', 0)
            else:
                val_acc = test_acc = epochs = 0
            
            data.append({
                'Model': model_type.replace('_', ' ').title(),
                'CV Mean': f"{cv_mean:.4f}",
                'CV Std': f"{cv_std:.4f}",
                'Val Accuracy': f"{val_acc:.4f}",
                'Test Accuracy': f"{test_acc:.4f}",
                'Epochs': int(epochs),
                'Parameters': self._estimate_parameters(model_type)
            })
        
        # Add ensemble row
        ensemble_mean = np.mean(self.cv_scores['ensemble'])
        ensemble_std = np.std(self.cv_scores['ensemble'])
        
        data.append({
            'Model': 'Ensemble (Soft Voting)',
            'CV Mean': f"{ensemble_mean:.4f}",
            'CV Std': f"{ensemble_std:.4f}",
            'Val Accuracy': '-',
            'Test Accuracy': f"{self.final_metrics.get('accuracy', 0):.4f}",
            'Epochs': '-',
            'Parameters': '4x Combined'
        })
        
        df = pd.DataFrame(data)
        
        # Create figure with table
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=df.values, colLabels=df.columns,
                        cellLoc='center', loc='center',
                        colWidths=[0.18, 0.12, 0.12, 0.14, 0.14, 0.12, 0.18])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.2)
        
        # Style header
        for i in range(len(df.columns)):
            table[(0, i)].set_facecolor('#4ECDC4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color rows
        colors = ['#FFE5E5', '#E5F9F7', '#E5F0F7', '#FFF0E5', '#E5F9F5']
        for i in range(1, len(df) + 1):
            for j in range(len(df.columns)):
                if i < len(df):
                    table[(i, j)].set_facecolor(colors[i-1])
                else:
                    table[(i, j)].set_facecolor('#D4F4DD')
        
        plt.title('Deep Learning Models Training Efficiency Analysis',
                 fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}training_efficiency_table.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Training Efficiency Table saved")
        return df
    
    def plot_confusion_matrices(self, figsize=(16, 10)):
        """
        Plot confusion matrices for all models + ensemble on final test set
        """
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        # Get individual model confusion matrices
        for i, model_info in enumerate(self.model_info):
            if i >= 4:
                break
            
            ax = axes[i]
            model_name = model_info['type']
            
            # Calculate confusion matrix for this model on test set
            cm = self._get_model_confusion_matrix(i)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       cbar_kws={'label': 'Count'}, annot_kws={'size': 12})
            
            test_acc = model_info.get('test_accuracy', 0)
            ax.set_title(f'{model_name.replace("_", " ").title()}\n(Test Acc: {test_acc:.4f})',
                        fontsize=12, fontweight='bold')
            ax.set_ylabel('True Label', fontweight='bold')
            ax.set_xlabel('Predicted Label', fontweight='bold')
            ax.set_xticklabels(['Benign', 'Malicious'])
            ax.set_yticklabels(['Benign', 'Malicious'])
        
        # Plot final ensemble confusion matrix
        ax = axes[4]
        sns.heatmap(self.final_confusion_matrix, annot=True, fmt='.0f', 
                   cmap='Greens', ax=ax, cbar_kws={'label': 'Count'},
                   annot_kws={'size': 12})
        
        final_acc = self.final_metrics.get('accuracy', 0)
        ax.set_title(f'Ensemble (Final Test)\n(Test Acc: {final_acc:.4f})',
                    fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontweight='bold')
        ax.set_xlabel('Predicted Label', fontweight='bold')
        ax.set_xticklabels(['Benign', 'Malicious'])
        ax.set_yticklabels(['Benign', 'Malicious'])
        
        # Hide last subplot
        axes[5].axis('off')
        
        plt.suptitle('Confusion Matrices – Final Test Set',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}confusion_matrices_test.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Confusion Matrices saved")
    
    def create_cv_vs_test_comparison_table(self):
        """
        Create comprehensive comparison table: CV vs Final Test
        """
        data = []
        
        model_types = list(self.cv_scores['individual'].keys())
        
        for model_type in model_types:
            cv_scores = self.cv_scores['individual'][model_type]
            cv_mean = np.mean(cv_scores)
            cv_std = np.std(cv_scores)
            cv_min = np.min(cv_scores)
            cv_max = np.max(cv_scores)
            
            # Find test accuracy
            test_acc = 0
            for info in self.model_info:
                if info['type'] == model_type:
                    test_acc = info.get('test_accuracy', 0)
                    break
            
            diff = abs(cv_mean - test_acc)
            
            data.append({
                'Model': model_type.replace('_', ' ').title(),
                'CV Mean': f"{cv_mean:.4f}",
                'CV Std': f"{cv_std:.4f}",
                'CV Min': f"{cv_min:.4f}",
                'CV Max': f"{cv_max:.4f}",
                'Test Acc': f"{test_acc:.4f}",
                'Difference': f"{diff:.4f}",
                'Gap %': f"{diff*100:.2f}%"
            })
        
        # Add ensemble
        cv_ensemble = self.cv_scores['ensemble']
        ens_cv_mean = np.mean(cv_ensemble)
        ens_cv_std = np.std(cv_ensemble)
        ens_cv_min = np.min(cv_ensemble)
        ens_cv_max = np.max(cv_ensemble)
        test_acc_ens = self.final_metrics.get('accuracy', 0)
        diff_ens = abs(ens_cv_mean - test_acc_ens)
        
        data.append({
            'Model': 'Ensemble',
            'CV Mean': f"{ens_cv_mean:.4f}",
            'CV Std': f"{ens_cv_std:.4f}",
            'CV Min': f"{ens_cv_min:.4f}",
            'CV Max': f"{ens_cv_max:.4f}",
            'Test Acc': f"{test_acc_ens:.4f}",
            'Difference': f"{diff_ens:.4f}",
            'Gap %': f"{diff_ens*100:.2f}%"
        })
        
        df = pd.DataFrame(data)
        
        # Create figure with table
        fig, ax = plt.subplots(figsize=(16, 4.5))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=df.values, colLabels=df.columns,
                        cellLoc='center', loc='center',
                        colWidths=[0.15, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12])
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2.4)
        
        # Style header
        for i in range(len(df.columns)):
            table[(0, i)].set_facecolor('#45B7D1')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color rows
        colors = ['#FFE5E5', '#E5F9F7', '#E5F0F7', '#FFF0E5', '#E5F9F5']
        for i in range(1, len(df) + 1):
            for j in range(len(df.columns)):
                if i < len(df):
                    table[(i, j)].set_facecolor(colors[i-1])
                else:
                    table[(i, j)].set_facecolor('#D4F4DD')
        
        plt.title('Cross-Validation vs Final Test Performance Comparison',
                 fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}cv_vs_test_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ CV vs Test Comparison Table saved")
        return df
    
    def plot_metrics_comparison(self, figsize=(14, 8)):
        """
        Compare all metrics (precision, recall, f1, etc.) across models
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        model_types = list(self.cv_scores['individual'].keys()) + ['Ensemble']
        
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
        metrics_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        x = np.arange(len(model_types))
        width = 0.2
        
        for idx, (metric, label) in enumerate(zip(metrics_to_plot, metrics_labels)):
            values = []
            
            for model_type in model_types:
                if model_type == 'Ensemble':
                    if metric in self.final_metrics:
                        val = self.final_metrics[metric]
                        if isinstance(val, dict):
                            val = val.get('mean', 0)
                        values.append(val)
                    else:
                        values.append(0)
                else:
                    # Get from CV metrics (average)
                    metric_vals = []
                    for fold_metrics in self.cv_metrics:
                        if metric in fold_metrics:
                            val = fold_metrics[metric]
                            if isinstance(val, dict):
                                val = val.get('mean', 0)
                            metric_vals.append(val)
                    values.append(np.mean(metric_vals) if metric_vals else 0)
            
            offset = width * (idx - 1.5)
            ax.bar(x + offset, values, width, label=label, alpha=0.8)
        
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_title('Performance Metrics Comparison Across Models',
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(model_types)
        ax.legend(loc='lower right', fontsize=11)
        ax.set_ylim([0.8, 1.02])
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}metrics_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Metrics Comparison Plot saved")
    
    def plot_cv_variance_analysis(self, figsize=(12, 8)):
        """
        Analyze variance across CV folds
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        model_types = list(self.cv_scores['individual'].keys())
        
        for idx, model_type in enumerate(model_types):
            ax = axes[idx]
            scores = self.cv_scores['individual'][model_type]
            
            # Violin plot
            parts = ax.violinplot([scores], positions=[1], widths=0.7,
                                 showmeans=True, showmedians=True)
            
            # Box plot overlay
            bp = ax.boxplot([scores], positions=[1], widths=0.3,
                           patch_artist=True, showfliers=True)
            
            for patch in bp['boxes']:
                patch.set_facecolor('#4ECDC4')
                patch.set_alpha(0.7)
            
            # Scatter points
            y = scores
            x = np.random.normal(1, 0.02, size=len(y))
            ax.scatter(x, y, alpha=0.5, s=100, color='#FF6B6B', zorder=3)
            
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            
            ax.set_ylabel('Accuracy', fontweight='bold')
            ax.set_title(f'{model_type.replace("_", " ").title()}\n'
                        f'Mean: {mean_score:.4f} ± {std_score:.4f}',
                        fontweight='bold', fontsize=11)
            ax.set_xticks([])
            ax.set_ylim([0.95, 1.005])
            ax.grid(axis='y', alpha=0.3)
        
        plt.suptitle('Cross-Validation Variance Analysis',
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}cv_variance_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ CV Variance Analysis saved")
    
    def _estimate_parameters(self, model_type):
        """Estimate number of parameters"""
        if model_type == 'base':
            return '~150K'
        elif model_type == 'multi_cnn':
            return '~145K'
        elif model_type == 'attention':
            return '~165K'
        else:  # wide
            return '~180K'
    
    def _get_model_confusion_matrix(self, model_idx):
        """Get confusion matrix for specific model on test set"""
        model = self.classifier.models[model_idx]
        
        # Get test data
        X_url_test = self.classifier.final_X_url_test
        X_num_test = self.classifier.final_X_num_test_scaled
        y_test = self.classifier.final_y_test
        
        # Prepare inputs
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        seq_data = self.classifier.tokenizer.texts_to_sequences(X_url_test)
        X_url_pad = pad_sequences(seq_data, maxlen=self.classifier.max_len,
                                 padding="post", truncating="post")
        
        # Predict
        pred_proba = model.predict(
            {"url_input": X_url_pad, "num_input": X_num_test},
            verbose=0, batch_size=512
        ).flatten()
        
        pred = (pred_proba > 0.5).astype(int)
        
        # Calculate confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, pred)
        
        return cm
    
    def generate_comprehensive_report(self):
        """Generate all visualizations and tables"""
        print("\n" + "="*80)
        print("📊 GENERATING COMPREHENSIVE REPORT")
        print("="*80)
        
        print("\n1️⃣  Generating CV Accuracies Box Plot...")
        self.plot_cv_accuracies_boxplot()
        
        print("\n2️⃣  Generating CV Accuracies Line Plot...")
        self.plot_cv_accuracies_line()
        
        print("\n3️⃣  Creating Training Efficiency Table...")
        efficiency_df = self.create_training_efficiency_table()
        
        print("\n4️⃣  Plotting Confusion Matrices...")
        self.plot_confusion_matrices()
        
        print("\n5️⃣  Creating CV vs Test Comparison Table...")
        comparison_df = self.create_cv_vs_test_comparison_table()
        
        print("\n6️⃣  Plotting Metrics Comparison...")
        self.plot_metrics_comparison()
        
        print("\n7️⃣  Analyzing CV Variance...")
        self.plot_cv_variance_analysis()
        
        print("\n" + "="*80)
        print("✅ ALL REPORTS GENERATED SUCCESSFULLY")
        print("="*80)
        print(f"\n📁 Reports saved to: {self.save_dir}")
        
        return {
            'efficiency_table': efficiency_df,
            'comparison_table': comparison_df
        }


# ==================== INTEGRATION FUNCTION ====================

def generate_final_comprehensive_reports(classifier):
    """
    Main function to generate all reports
    Call this at the end of main()
    """
    reporter = ComprehensiveReporting(classifier, save_dir="reports/")
    results = reporter.generate_comprehensive_report()
    return reporter, results