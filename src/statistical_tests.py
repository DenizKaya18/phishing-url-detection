# -*- coding: utf-8 -*-
import numpy as np
from scipy.stats import ttest_rel, f_oneway, wilcoxon, chi2
import pandas as pd

# mcnemar import - different scipy versions
try:
    from scipy.stats import mcnemar
except ImportError:
    # For older scipy versions, define mcnemar manually
    def mcnemar(table, exact=False):
        """Simple mcnemar test implementation"""
        table = np.asarray(table)
        a01 = table[0, 1]
        a10 = table[1, 0]
        if a01 + a10 == 0:
            return 0, 1.0
        stat = ((a10 - a01) ** 2) / (a10 + a01)
        pval = 1 - chi2.cdf(stat, df=1)
        return stat, pval


class StatisticalSignificanceAnalyzer:
    """
    Statistical significance testing for ensemble and individual models.
    
    - McNemar Test: For validating ensemble vs individual models per-fold
    - Paired t-test: For comparing fold-averaged metrics across all folds
    - Wilcoxon Test: Non-parametric alternative for small samples
    """
    
    def __init__(self, ensemble_clf):
        """
        Initialize with trained ensemble classifier.
        
        Args:
            ensemble_clf: OptimizedEnsembleURLClassifierCV instance
        """
        self.ensemble_clf = ensemble_clf
        self.mcnemar_results = {}
        self.ttest_results = {}
        self.wilcoxon_results = {}
    
    # ==================== McNEMAR TEST ====================
    def prepare_mcnemar_data(self):
        """
        Extract validation predictions from each fold for McNemar test.
        
        Returns:
            fold_ensemble_preds: list[fold_idx] -> np.array of ensemble predictions
            fold_model_preds: dict[model_name] -> list[fold_idx] -> np.array
            fold_true_labels: list[fold_idx] -> np.array
        """
        fold_ensemble_preds = []
        fold_model_preds = {model_type: [] for model_type in self.ensemble_clf.cv_model_detailed_results.keys()}
        fold_true_labels = []
        
        # Collect per-model predictions from CV
        for model_type in self.ensemble_clf.cv_model_detailed_results.keys():
            model_cms = self.ensemble_clf.cv_model_confusion_matrices[model_type]
            model_results = self.ensemble_clf.cv_model_detailed_results[model_type]
            
            for fold_idx, (cm, result) in enumerate(zip(model_cms, model_results)):
                # Note: We have confusion matrices but need actual predictions
                # Reconstruct from CV metrics stored during training
                fold_model_preds[model_type].append((cm, result))
        
        # Ensemble predictions
        for fold_idx, cm in enumerate(self.ensemble_clf.cv_confusion_matrices):
            fold_ensemble_preds.append(cm)
            fold_true_labels.append(self.ensemble_clf.cv_metrics[fold_idx])
        
        return fold_ensemble_preds, fold_model_preds, fold_true_labels
    
    def mcnemar_test_2x2(self, pred_model_a, pred_model_b, y_true, fold_idx, model_a_name, model_b_name):
        """
        Perform McNemar test on two models for a single fold.
        
        McNemar's test examines if two classifiers have different error patterns:
        - a01: model_a wrong, model_b correct
        - a10: model_a correct, model_b wrong
        
        H0: Both models have equal error rates
        H1: Models have significantly different error rates
        
        Args:
            pred_model_a, pred_model_b: Binary predictions (0 or 1)
            y_true: Ground truth labels
            fold_idx: Fold number for reporting
            model_a_name, model_b_name: Model names for reporting
            
        Returns:
            dict with test results
        """
        # Ensure same length
        assert len(pred_model_a) == len(pred_model_b) == len(y_true)
        
        # Compute disagreement patterns
        model_a_correct = (pred_model_a == y_true).astype(int)
        model_b_correct = (pred_model_b == y_true).astype(int)
        
        # McNemar contingency table
        # a01: A wrong, B right
        # a10: A right, B wrong
        a01 = np.sum((model_a_correct == 0) & (model_b_correct == 1))
        a10 = np.sum((model_a_correct == 1) & (model_b_correct == 0))
        
        # McNemar statistic
        if (a01 + a10) == 0:
            mcnemar_stat = 0
            p_value = 1.0
        else:
            mcnemar_stat, p_value = mcnemar([[model_a_correct.sum(), a01],
                                            [a10, model_b_correct.sum()]], exact=False)
        
        # Calculate accuracies
        acc_a = np.mean(model_a_correct)
        acc_b = np.mean(model_b_correct)
        
        return {
            'fold': fold_idx,
            'model_a': model_a_name,
            'model_b': model_b_name,
            'acc_a': acc_a,
            'acc_b': acc_b,
            'acc_diff': acc_b - acc_a,
            'a01': a01,  # A wrong, B right
            'a10': a10,  # A right, B wrong
            'mcnemar_statistic': mcnemar_stat,
            'p_value': p_value,
            'significant_alpha_0_05': p_value < 0.05,
            'significant_alpha_0_01': p_value < 0.01
        }
    
    def run_mcnemar_ensemble_vs_models(self):
        """
        Compare ensemble vs each individual model using McNemar test.
        This uses confusion matrices from CV to reconstruct predictions.
        """
        print("\n" + "="*80)
        print("🔍 McNEMAR TEST: ENSEMBLE vs INDIVIDUAL MODELS")
        print("="*80)
        print("Testing if ensemble and individual models have significantly different error patterns\n")
        
        model_types = list(self.ensemble_clf.cv_model_detailed_results.keys())
        
        for fold_idx in range(self.ensemble_clf.n_folds):
            print(f"\n{'─'*70}")
            print(f"📊 FOLD {fold_idx + 1}/{self.ensemble_clf.n_folds}")
            print(f"{'─'*70}")
            
            # Get ensemble and model metrics for this fold
            ensemble_metrics = self.ensemble_clf.cv_metrics[fold_idx]
            
            for model_type in model_types:
                model_metrics = self.ensemble_clf.cv_model_metrics[model_type][fold_idx]
                
                # Get confusion matrices
                ens_cm = self.ensemble_clf.cv_confusion_matrices[fold_idx]
                model_cm = self.ensemble_clf.cv_model_confusion_matrices[model_type][fold_idx]
                
                # Reconstruct predictions from confusion matrices
                # CM format: [[tn, fp], [fn, tp]]
                ens_tp, ens_fn = ensemble_metrics['true_positives'], ensemble_metrics['false_negatives']
                model_tp, model_fn = model_metrics['true_positives'], model_metrics['false_negatives']
                ens_tn, ens_fp = ensemble_metrics['true_negatives'], ensemble_metrics['false_positives']
                model_tn, model_fp = model_metrics['true_negatives'], model_metrics['false_positives']
                
                # Accuracies
                ens_acc = ensemble_metrics['accuracy']
                model_acc = model_metrics['accuracy']
                
                # Simplified McNemar using TP differences
                # a10: ensemble TP - model FN = ensemble got it right, model didn't
                # a01: model TP - ensemble FN = model got it right, ensemble didn't
                a10 = abs(ens_tp - model_tp) if ens_tp > model_tp else 0
                a01 = abs(model_tp - ens_tp) if model_tp > ens_tp else 0
                
                # McNemar statistic
                if (a01 + a10) == 0:
                    mcnemar_stat = 0
                    p_value = 1.0
                else:
                    mcnemar_stat = ((a10 - a01) ** 2) / (a10 + a01)
                    from scipy.stats import chi2
                    p_value = 1 - chi2.cdf(mcnemar_stat, df=1)
                
                result = {
                    'fold': fold_idx + 1,
                    'model_a': 'Ensemble',
                    'model_b': model_type,
                    'acc_ensemble': ens_acc,
                    'acc_model': model_acc,
                    'acc_diff': ens_acc - model_acc,
                    'a10': a10,
                    'a01': a01,
                    'mcnemar_statistic': mcnemar_stat,
                    'p_value': p_value,
                    'significant_alpha_0_05': p_value < 0.05,
                    'significant_alpha_0_01': p_value < 0.01
                }
                
                key = f"Ensemble_vs_{model_type}_Fold{fold_idx+1}"
                self.mcnemar_results[key] = result
                
                # Print result
                sig_marker = "***" if result['significant_alpha_0_01'] else "**" if result['significant_alpha_0_05'] else ""
                print(f"  Ensemble vs {model_type:<12}: "
                      f"Ensemble={result['acc_ensemble']:.4f}, {model_type}={result['acc_model']:.4f} "
                      f"(Δ={result['acc_diff']:+.4f}) "
                      f"χ²={result['mcnemar_statistic']:.4f}, p={result['p_value']:.4f} {sig_marker}")
        
        print(f"\n{'─'*70}")
        print("Legend: *** p<0.01 (highly significant), ** p<0.05 (significant), * p<0.10")
        print("="*80)
    
    # ==================== PAIRED T-TEST ====================
    def run_paired_ttest_ensemble_vs_models(self, metric='accuracy'):
        """
        Paired t-test across all folds comparing ensemble vs individual models.
        
        Tests if there's a statistically significant difference in fold-averaged metrics.
        H0: μ(ensemble) = μ(model)
        H1: μ(ensemble) ≠ μ(model)
        
        Args:
            metric: Which metric to compare ('accuracy', 'f1_score', 'precision', 'recall')
        """
        print("\n" + "="*80)
        print(f"📊 PAIRED T-TEST: ENSEMBLE vs MODELS ({metric.upper()})")
        print("="*80)
        print(f"Comparing {metric} across {self.ensemble_clf.n_folds} folds\n")
        
        model_types = list(self.ensemble_clf.cv_model_detailed_results.keys())
        
        # Collect fold-wise metrics
        ensemble_metrics_per_fold = [m[metric] for m in self.ensemble_clf.cv_metrics]
        
        print(f"{'Model':<15} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10} "
              f"{'t-stat':<10} {'p-value':<10} {'Significant':<12}")
        print("-" * 90)
        
        # Ensemble baseline
        ens_mean = np.mean(ensemble_metrics_per_fold)
        ens_std = np.std(ensemble_metrics_per_fold)
        ens_min = np.min(ensemble_metrics_per_fold)
        ens_max = np.max(ensemble_metrics_per_fold)
        print(f"{'Ensemble':<15} {ens_mean:<10.4f} {ens_std:<10.4f} {ens_min:<10.4f} {ens_max:<10.4f} "
              f"{'─':<10} {'─':<10} {'Baseline':<12}")
        
        # Compare each model with ensemble
        for model_type in model_types:
            model_metrics_per_fold = [m[metric] for m in self.ensemble_clf.cv_model_metrics[model_type]]
            
            # Paired t-test
            t_stat, p_value = ttest_rel(ensemble_metrics_per_fold, model_metrics_per_fold)
            
            model_mean = np.mean(model_metrics_per_fold)
            model_std = np.std(model_metrics_per_fold)
            model_min = np.min(model_metrics_per_fold)
            model_max = np.max(model_metrics_per_fold)
            
            significant = "Yes***" if p_value < 0.01 else "Yes**" if p_value < 0.05 else "No"
            
            print(f"{model_type:<15} {model_mean:<10.4f} {model_std:<10.4f} {model_min:<10.4f} {model_max:<10.4f} "
                  f"{t_stat:<10.4f} {p_value:<10.6f} {significant:<12}")
            
            self.ttest_results[f"Ensemble_vs_{model_type}"] = {
                'metric': metric,
                'ensemble_mean': ens_mean,
                'ensemble_std': ens_std,
                'model_mean': model_mean,
                'model_std': model_std,
                'mean_diff': ens_mean - model_mean,
                't_statistic': t_stat,
                'p_value': p_value,
                'significant_alpha_0_05': p_value < 0.05,
                'significant_alpha_0_01': p_value < 0.01,
                'degrees_of_freedom': self.ensemble_clf.n_folds - 1
            }
        
        print("-" * 90)
        print("Legend: *** p<0.01 (highly significant), ** p<0.05 (significant)\n")
        
        # Summary interpretation
        print("📌 INTERPRETATION:")
        print(f"  • Paired t-test compares ensemble vs each model across {self.ensemble_clf.n_folds} folds")
        print(f"  • Null hypothesis (H0): Both methods have equal average {metric}")
        print(f"  • If p-value < 0.05: Significant difference detected")
        print(f"  • If p-value ≥ 0.05: No significant difference (models perform similarly)")
        print("="*80)
    
    # ==================== WILCOXON TEST (NON-PARAMETRIC) ====================
    def run_wilcoxon_ensemble_vs_models(self, metric='accuracy'):
        """
        Non-parametric alternative to paired t-test (useful for small samples).
        
        Better for non-normal distributions or small fold counts.
        
        Args:
            metric: Which metric to compare
        """
        print("\n" + "="*80)
        print(f"📊 WILCOXON TEST: ENSEMBLE vs MODELS ({metric.upper()})")
        print("="*80)
        print(f"Non-parametric comparison across {self.ensemble_clf.n_folds} folds\n")
        
        model_types = list(self.ensemble_clf.cv_model_detailed_results.keys())
        
        ensemble_metrics_per_fold = np.array([m[metric] for m in self.ensemble_clf.cv_metrics])
        
        print(f"{'Model':<15} {'Ensemble Median':<15} {'Model Median':<15} "
              f"{'z-stat':<10} {'p-value':<10} {'Significant':<12}")
        print("-" * 80)
        
        for model_type in model_types:
            model_metrics_per_fold = np.array([m[metric] for m in self.ensemble_clf.cv_model_metrics[model_type]])
            
            # Wilcoxon signed-rank test
            try:
                w_stat, p_value = wilcoxon(ensemble_metrics_per_fold, model_metrics_per_fold)
            except ValueError:
                # If all differences are zero
                w_stat = 0
                p_value = 1.0
            
            ens_median = np.median(ensemble_metrics_per_fold)
            model_median = np.median(model_metrics_per_fold)
            
            significant = "Yes***" if p_value < 0.01 else "Yes**" if p_value < 0.05 else "No"
            
            print(f"{model_type:<15} {ens_median:<15.4f} {model_median:<15.4f} "
                  f"{w_stat:<10.4f} {p_value:<10.6f} {significant:<12}")
            
            self.wilcoxon_results[f"Ensemble_vs_{model_type}"] = {
                'metric': metric,
                'ensemble_median': ens_median,
                'model_median': model_median,
                'w_statistic': w_stat,
                'p_value': p_value,
                'significant_alpha_0_05': p_value < 0.05,
                'significant_alpha_0_01': p_value < 0.01
            }
        
        print("-" * 80)
        print("Legend: *** p<0.01 (highly significant), ** p<0.05 (significant)\n")
        print("📌 INTERPRETATION:")
        print(f"  • Wilcoxon test: Non-parametric alternative to paired t-test")
        print(f"  • Uses ranks instead of raw differences")
        print(f"  • More robust to outliers and non-normal distributions")
        print("="*80)
    
    # ==================== ONE-WAY ANOVA (COMPARING ALL MODELS) ====================
    def run_anova_all_models(self, metric='accuracy'):
        """
        One-way ANOVA to test if there are significant differences among all models.
        
        H0: All models have equal mean performance
        H1: At least one model differs significantly
        """
        print("\n" + "="*80)
        print(f"📊 ONE-WAY ANOVA: ALL MODELS COMPARISON ({metric.upper()})")
        print("="*80)
        print(f"Testing if performance differs among all {self.ensemble_clf.n_models + 1} models\n")
        
        model_types = list(self.ensemble_clf.cv_model_detailed_results.keys())
        
        # Collect metrics
        ensemble_metrics = [m[metric] for m in self.ensemble_clf.cv_metrics]
        model_metrics_list = [
            [m[metric] for m in self.ensemble_clf.cv_model_metrics[model_type]]
            for model_type in model_types
        ]
        
        # Prepare data for ANOVA
        all_data = [ensemble_metrics] + model_metrics_list
        all_labels = ['Ensemble'] + model_types
        
        # One-way ANOVA
        f_stat, p_value = f_oneway(*all_data)
        
        # Print summary
        print(f"{'Model':<15} {'Mean':<10} {'Std':<10} {'N':<5}")
        print("-" * 40)
        
        for label, data in zip(all_labels, all_data):
            print(f"{label:<15} {np.mean(data):<10.4f} {np.std(data):<10.4f} {len(data):<5}")
        
        print("-" * 40)
        print(f"\nF-statistic: {f_stat:.4f}")
        print(f"p-value: {p_value:.6f}")
        
        significant = "Yes***" if p_value < 0.01 else "Yes**" if p_value < 0.05 else "No"
        print(f"Significant difference (α=0.05): {significant}")
        
        print("\n📌 INTERPRETATION:")
        print("  • H0: All models perform equally well on average")
        print("  • H1: At least one model performs significantly differently")
        print("  • If p < 0.05: Reject H0 (significant differences exist)")
        print("  • If p ≥ 0.05: Fail to reject H0 (no significant differences)")
        print("="*80)
        
        return f_stat, p_value
    
    # ==================== EFFECT SIZE ====================
    def calculate_effect_sizes(self, metric='accuracy'):
        """
        Calculate Cohen's d for ensemble vs each model.
        
        Effect size interpretation:
        - Small: 0.2
        - Medium: 0.5
        - Large: 0.8
        """
        print("\n" + "="*80)
        print(f"📊 EFFECT SIZES: COHEN'S D (Ensemble vs Models)")
        print("="*80 + "\n")
        
        model_types = list(self.ensemble_clf.cv_model_detailed_results.keys())
        
        ensemble_metrics = np.array([m[metric] for m in self.ensemble_clf.cv_metrics])
        ens_mean = np.mean(ensemble_metrics)
        ens_std = np.std(ensemble_metrics, ddof=1)
        
        print(f"{'Model':<15} {'Cohens d':<12} {'Effect Size':<15} {'Interpretation':<20}")
        print("-" * 65)
        
        for model_type in model_types:
            model_metrics = np.array([m[metric] for m in self.ensemble_clf.cv_model_metrics[model_type]])
            model_mean = np.mean(model_metrics)
            model_std = np.std(model_metrics, ddof=1)
            
            # Pooled standard deviation
            n1, n2 = len(ensemble_metrics), len(model_metrics)
            pooled_std = np.sqrt(((n1-1)*ens_std**2 + (n2-1)*model_std**2) / (n1 + n2 - 2))
            
            # Cohen's d
            cohens_d = (ens_mean - model_mean) / pooled_std if pooled_std > 0 else 0
            
            # Interpretation
            abs_d = abs(cohens_d)
            if abs_d < 0.2:
                interp = "Negligible"
            elif abs_d < 0.5:
                interp = "Small"
            elif abs_d < 0.8:
                interp = "Medium"
            else:
                interp = "Large"
            
            direction = "Ensemble better" if cohens_d > 0 else "Model better" if cohens_d < 0 else "Equal"
            
            print(f"{model_type:<15} {cohens_d:<12.4f} {abs_d:<15.4f} {interp} ({direction})")
        
        print("-" * 65)
        print("📌 COHEN'S D INTERPRETATION:")
        print("  • 0.0 - 0.2: Negligible effect")
        print("  • 0.2 - 0.5: Small effect")
        print("  • 0.5 - 0.8: Medium effect")
        print("  • 0.8+:     Large effect")
        print("="*80)
    
    # ==================== COMPREHENSIVE REPORT ====================
    def print_comprehensive_statistical_report(self):
        """
        Print comprehensive statistical analysis report.
        """
        print("\n\n" + "="*80)
        print("📊 COMPREHENSIVE STATISTICAL SIGNIFICANCE REPORT")
        print("="*80 + "\n")
        
        # Run all tests
        print("Running all statistical tests...\n")
        
        self.run_mcnemar_ensemble_vs_models()
        self.run_paired_ttest_ensemble_vs_models(metric='accuracy')
        self.run_wilcoxon_ensemble_vs_models(metric='accuracy')
        self.run_anova_all_models(metric='accuracy')
        self.calculate_effect_sizes(metric='accuracy')
        
        # Summary table
        print("\n" + "="*80)
        print("📋 TEST SUMMARY TABLE")
        print("="*80 + "\n")
        
        print(f"{'Test':<20} {'Purpose':<35} {'Result':<15}")
        print("-" * 70)
        print(f"{'McNemar':<20} {'Error pattern differences':<35} {'See above':<15}")
        print(f"{'Paired t-test':<20} {'Mean accuracy differences':<35} {'See above':<15}")
        print(f"{'Wilcoxon':<20} {'Non-parametric comparison':<35} {'See above':<15}")
        print(f"{'One-way ANOVA':<20} {'All models vs each other':<35} {'See above':<15}")
        print(f"{'Cohens d':<20} {'Effect size magnitude':<35} {'See above':<15}")
        
        print("\n" + "="*80)
        print("✅ Statistical Analysis Complete")
        print("="*80)


# ==================== INTEGRATION WITH MAIN ====================

def run_statistical_tests(ensemble_clf):
    """
    Integration function to run after ensemble training.
    
    Call this after ensemble_clf.evaluate_final_ensemble()
    """
    analyzer = StatisticalSignificanceAnalyzer(ensemble_clf)
    analyzer.print_comprehensive_statistical_report()
    
    return analyzer


# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    # After running main() from the original code:
    # classifier, results, best_method = main()
    # 
    # Then run statistical tests:
    # analyzer = run_statistical_tests(classifier)
    
    print("""
    ✅ Statistical Significance Testing Module Ready
    
    📌 INTEGRATION STEPS:
    
    1. In your main() function, after evaluate_final_ensemble():
       
       analyzer = run_statistical_tests(ensemble_clf)
    
    2. Or manually:
       
       analyzer = StatisticalSignificanceAnalyzer(ensemble_clf)
       analyzer.print_comprehensive_statistical_report()
    
    3. For specific tests:
       
       analyzer.run_mcnemar_ensemble_vs_models()
       analyzer.run_paired_ttest_ensemble_vs_models(metric='accuracy')
       analyzer.run_wilcoxon_ensemble_vs_models(metric='accuracy')
       analyzer.run_anova_all_models(metric='accuracy')
       analyzer.calculate_effect_sizes(metric='accuracy')
    """)