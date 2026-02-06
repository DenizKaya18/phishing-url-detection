# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, f_oneway, wilcoxon, chi2
from .config import RESULTS_DIR
import os

# Manual McNemar implementation for cross-version compatibility
def mcnemar_test_logic(table):
    """
    Performs McNemar's test on a 2x2 contingency table.
    Used to determine if there is a significant difference between error rates.
    """
    table = np.asarray(table)
    a01 = table[0, 1]  # Model A correct, Model B wrong
    a10 = table[1, 0]  # Model A wrong, Model B correct
    
    if a01 + a10 == 0:
        return 0.0, 1.0
    
    # Chi-squared statistic calculation
    statistic = (abs(a10 - a01) - 1)**2 / (a10 + a01)
    p_value = 1 - chi2.cdf(statistic, df=1)
    return statistic, p_value

class StatisticalSignificanceAnalyzer:
    """
    Statistical significance testing suite for ensemble vs individual models.
    """
    def __init__(self, ensemble_clf):
        self.ensemble_clf = ensemble_clf
        self.results_path = os.path.join(RESULTS_DIR, "statistical_analysis_report.csv")

    def run_mcnemar_analysis(self):
        print("\n" + "="*80)
        print("🔍 McNEMAR TEST: ENSEMBLE vs INDIVIDUAL MODELS")
        print("="*80)
        
        model_types = list(self.ensemble_clf.cv_model_metrics.keys())
        results_list = []

        for fold_idx in range(self.ensemble_clf.n_folds):
            print(f"\n--- FOLD {fold_idx + 1} ---")
            ens_metrics = self.ensemble_clf.cv_metrics[fold_idx]
            
            for m_type in model_types:
                m_metrics = self.ensemble_clf.cv_model_metrics[m_type][fold_idx]
                
                # a10: Ensemble got it right, Model got it wrong
                # a01: Model got it right, Ensemble got it wrong
                ens_tp = ens_metrics['true_positives']
                m_tp = m_metrics['true_positives']
                
                a10 = max(0, ens_tp - m_tp)
                a01 = max(0, m_tp - ens_tp)
                
                stat, p = mcnemar_test_logic([[0, a01], [a10, 0]])
                is_sig = p < 0.05
                sig_marker = "***" if p < 0.01 else "**" if p < 0.05 else ""
                
                print(f"   Ensemble vs {m_type:<12}: p={p:.4f} {sig_marker}")

                results_list.append({
                    'Fold': fold_idx + 1,
                    'Model': m_type,
                    'Ensemble_Acc': ens_metrics['accuracy'],
                    'Model_Acc': m_metrics['accuracy'],
                    'p_value': p,
                    'Significant': is_sig
                })

        return pd.DataFrame(results_list)

    def run_paired_ttest(self, metric='accuracy'):
        print("\n" + "="*80)
        print(f"📊 PAIRED T-TEST: ENSEMBLE vs MODELS ({metric.upper()})")
        print("="*80)
        
        ens_scores = [m[metric] for m in self.ensemble_clf.cv_metrics]
        model_types = list(self.ensemble_clf.cv_model_metrics.keys())
        
        print(f"{'Model':<15} {'Mean Diff':<12} {'t-stat':<10} {'p-value':<10} {'Sig.'}")
        print("-" * 65)
        
        for m_type in model_types:
            m_scores = [m[metric] for m in self.ensemble_clf.cv_model_metrics[m_type]]
            t_stat, p = ttest_rel(ens_scores, m_scores)
            
            sig = "Yes" if p < 0.05 else "No"
            diff = np.mean(ens_scores) - np.mean(m_scores)
            print(f"{m_type:<15} {diff:<12.4f} {t_stat:<10.4f} {p:<10.6f} {sig}")

    def run_wilcoxon_test(self, metric='accuracy'):
        print("\n" + "="*80)
        print(f"📊 WILCOXON SIGNED-RANK TEST ({metric.upper()})")
        print("="*80)
        
        ens_scores = np.array([m[metric] for m in self.ensemble_clf.cv_metrics])
        model_types = list(self.ensemble_clf.cv_model_metrics.keys())

        for m_type in model_types:
            m_scores = np.array([m[metric] for m in self.ensemble_clf.cv_model_metrics[m_type]])
            try:
                stat, p = wilcoxon(ens_scores, m_scores)
                sig = "Yes" if p < 0.05 else "No"
                print(f"Ensemble vs {m_type:<12}: p={p:.6f} (Significant: {sig})")
            except Exception as e:
                print(f"Ensemble vs {m_type:<12}: Error - {e}")

    def calculate_effect_size(self, metric='accuracy'):
        """Calculates Cohen's d for result magnitude."""
        print("\n" + "="*80)
        print(f"📏 EFFECT SIZE: COHEN'S D (Ensemble vs Models)")
        print("="*80)
        
        ens_scores = np.array([m[metric] for m in self.ensemble_clf.cv_metrics])
        model_types = list(self.ensemble_clf.cv_model_metrics.keys())
        
        for m_type in model_types:
            m_scores = np.array([m[metric] for m in self.ensemble_clf.cv_model_metrics[m_type]])
            
            s1, s2 = np.std(ens_scores, ddof=1), np.std(m_scores, ddof=1)
            n1, n2 = len(ens_scores), len(m_scores)
            pooled_sd = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
            
            d = (np.mean(ens_scores) - np.mean(m_scores)) / pooled_sd if pooled_sd != 0 else 0
            interp = "Large" if abs(d) > 0.8 else "Medium" if abs(d) > 0.5 else "Small"
            print(f"Ensemble vs {m_type:<12}: d={d:.4f} ({interp} effect)")

    def generate_full_report(self):
        """Executes all statistical tests and saves the results."""
        mcnemar_df = self.run_mcnemar_analysis()
        self.run_paired_ttest()
        self.run_wilcoxon_test()
        self.calculate_effect_size()
        
        mcnemar_df.to_csv(self.results_path, index=False)
        print(f"\n✓ Full statistical significance report saved to: {self.results_path}")

# --- BRIDGE FUNCTION FOR MAIN.PY ---
def run_statistical_analysis(ensemble_clf):
    """
    Main entry point called by the execution pipeline.
    """
    analyzer = StatisticalSignificanceAnalyzer(ensemble_clf)
    analyzer.generate_full_report()
    return analyzer