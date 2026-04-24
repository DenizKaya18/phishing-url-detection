# -*- coding: utf-8 -*-
"""
Report Generation Module
"""

import os
import shutil
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class ReportGenerator:
    """Generates reports and visualizations"""
    
    def __init__(self, config):
        self.config = config
    
    def safe_copy_to_drive(self, local_path, fname=None, max_retries=3):
        """Safely copy file to output folder with retries"""
        try:
            if not fname:
                fname = os.path.basename(local_path)
            dst = os.path.join(self.config.output_folder, fname)
            
            if not os.path.exists(self.config.output_folder):
                try:
                    os.makedirs(self.config.output_folder, exist_ok=True)
                except Exception as e:
                    print(f"[WARNING] Failed to create output folder: {e}")
                    return False
            
            for attempt in range(1, max_retries + 1):
                try:
                    shutil.copy2(local_path, dst)
                    print(f"[OK] {fname} backed up to output: {dst}")
                    return True
                except Exception as e:
                    if attempt == max_retries:
                        print(f"[WARNING] Failed to copy {fname} to output: {e}")
                        return False
        except Exception as e:
            print(f"[WARNING] Error in safe_copy_to_drive: {e}")
            return False
    
    def save_report(self, df, filename, copy_to_drive=True):
        """Save report locally and optionally to output folder"""
        try:
            tmp = filename + ".tmp"
            df.to_csv(tmp, index=False)
            try:
                os.replace(tmp, filename)
            except Exception:
                shutil.move(tmp, filename)
            print(f"[OK] Saved locally: {filename}")
        except Exception as e:
            print(f"[ERROR] Failed to save {filename}: {e}")
            return False
        
        if copy_to_drive:
            return self.safe_copy_to_drive(filename, os.path.basename(filename))
        return True
    
    def generate_reports(self, full_report_data):
        """Generate all reports from training results"""
        print("\n" + "="*80)
        print("=== GENERATING REPORTS ===")
        print("="*80)
        
        df_report = pd.DataFrame(full_report_data)
        
        if df_report.empty:
            print("[WARNING] No data to generate reports")
            return
        
        # Ensure timing columns exist
        time_cols = ['Avg_Single_Pred_Time_ms', 'Std_Single_Pred_Time_ms']
        for c in time_cols:
            if c not in df_report.columns:
                df_report[c] = 0.0
            df_report[c] = pd.to_numeric(df_report[c], errors='coerce').fillna(0.0)
        
        # Round timing columns
        df_report['Avg_Single_Pred_Time_ms'] = df_report['Avg_Single_Pred_Time_ms'].round(6)
        df_report['Std_Single_Pred_Time_ms'] = df_report['Std_Single_Pred_Time_ms'].round(6)
        
        # 1. Detailed report per fold
        self.save_report(df_report, "Detailed_Performance_Report_Per_Fold.csv")
        
        # 2. Baseline average performance
        df_base = df_report[df_report['Scenario'] == 'BASELINE']
        if not df_base.empty:
            metrics_agg = [
                'Train_Time_Sec', 'Avg_Single_Pred_Time_ms', 'Std_Single_Pred_Time_ms',
                'Imbalance_Ratio', 'Accuracy', 'F1_Score', 'Precision', 'Recall',
                'Sensitivity', 'Specificity', 'FNR', 'FPR', 'TN', 'FP', 'FN', 'TP'
            ]
            valid_metrics = [c for c in metrics_agg if c in df_base.columns]
            
            if valid_metrics:
                avg_base = df_base.groupby('Model')[valid_metrics].mean().reset_index()
                num_cols = avg_base.select_dtypes(include=[np.number]).columns.tolist()
                avg_base[num_cols] = avg_base[num_cols].round(6)
                self.save_report(avg_base, "Baseline_Average_Performance.csv")
            
            # 3. Confusion matrices
            self._generate_confusion_matrices(df_base)
        
        # 4. Statistical analysis
        self._generate_statistical_report(df_base)
        
        print(f"\n=== REPORTS COMPLETED. Files saved to: {self.config.output_folder} ===")
    
    def _generate_confusion_matrices(self, df_base):
        """Generate confusion matrix visualizations"""
        cm_cols = ['TN', 'FP', 'FN', 'TP']
        if not all(c in df_base.columns for c in cm_cols):
            print("[WARNING] Confusion matrix columns missing, skipping CM plots")
            return
        
        avg_cm = df_base.groupby('Model')[cm_cols].mean()
        
        for model_name in avg_cm.index:
            try:
                tn, fp, fn, tp = avg_cm.loc[model_name].astype(float).values
                cm_arr = np.array([[tn, fp], [fn, tp]])
                
                plt.figure(figsize=(5, 4))
                sns.heatmap(
                    cm_arr, annot=True, fmt=".1f", cbar=False,
                    xticklabels=['Pred: Normal', 'Pred: Phishing'],
                    yticklabels=['True: Normal', 'True: Phishing']
                )
                plt.title(f"Avg CM - {model_name} (Baseline)")
                plt.ylabel("True")
                plt.xlabel("Pred")
                plt.tight_layout()
                
                img_name = f"CM_Baseline_{model_name.replace(' ', '_')}.png"
                plt.savefig(img_name, dpi=300)
                plt.close()
                
                self.safe_copy_to_drive(img_name, img_name)
            except Exception as e:
                print(f"[ERROR] Failed to generate CM for {model_name}: {e}")
    
    def _generate_statistical_report(self, df_base):
        """Generate comprehensive statistical significance report - Friedman + Nemenyi focus"""
        if df_base.empty:
            print("[WARNING] No baseline data for statistical analysis")
            return
        
        try:
            # Gerekli kütüphaneler
            import numpy as np
            import pandas as pd
            from scipy.stats import friedmanchisquare, chi2
            import scikit_posthocs as sp
            
            # Sonuçları model bazında topla
            results_per_model = {}
            for model in df_base['Model'].unique():
                model_data = df_base[df_base['Model'] == model].sort_values('Fold')
                accuracies = model_data['Accuracy'].values.tolist()
                
                results_per_model[model] = {
                    'fold_accuracies': accuracies,
                }
            
            model_names = list(results_per_model.keys())
            if not model_names:
                print("[WARNING] No models found for statistical analysis")
                return
            
            n_folds = len(results_per_model[model_names[0]]['fold_accuracies'])
            if any(len(results_per_model[m]['fold_accuracies']) != n_folds for m in model_names):
                print("[ERROR] Models have different number of folds → analysis aborted")
                return
            
            report_lines = []
            report_lines.append("=" * 80)
            report_lines.append("📊 COMPREHENSIVE STATISTICAL SIGNIFICANCE REPORT")
            report_lines.append("Friedman Test + Nemenyi Post-hoc - Classical ML Models")
            report_lines.append("=" * 80)
            report_lines.append(f"Folds: {n_folds} | Models: {len(model_names)}")
            report_lines.append("")
            
            # ────────────────────────────────────────────────
            # 1. Friedman Testi
            # ────────────────────────────────────────────────
            report_lines.append("=" * 80)
            report_lines.append("🔬 FRIEDMAN TEST (Non-parametric ANOVA alternative)")
            report_lines.append("=" * 80)
            report_lines.append("H0: All models perform equally (no difference in ranks)")
            report_lines.append("")
            
            # Fold × Model matrisi (her satır bir fold)
            data_matrix = np.array([results_per_model[m]['fold_accuracies'] for m in model_names]).T
            
            # Ortalama doğrulukları da gösterelim
            mean_acc = {m: np.mean(results_per_model[m]['fold_accuracies']) for m in model_names}
            report_lines.append("Model Performans Özeti (Mean Accuracy):")
            for m, acc in sorted(mean_acc.items(), key=lambda x: x[1], reverse=True):
                report_lines.append(f"  {m:<25} {acc:.4f}")
            report_lines.append("")
            
            stat, p = friedmanchisquare(*data_matrix.T)
            
            report_lines.append(f"Chi-square statistic: {stat:.4f}")
            report_lines.append(f"p-value:              {p:.6f}")
            
            sig_marker = "***" if p < 0.01 else "**" if p < 0.05 else ""
            result = f"Significant{sig_marker} (reject H0)" if p < 0.05 else "Not significant (fail to reject H0)"
            report_lines.append(f"Result (α=0.05):      {result}")
            report_lines.append("")
            report_lines.append("Interpretation:")
            report_lines.append("  • p < 0.05 → At least one model differs significantly")
            report_lines.append("  • p ≥ 0.05 → No evidence of overall difference")
            report_lines.append("")
            
            # ────────────────────────────────────────────────
            # 2. Nemenyi Post-hoc (sadece Friedman anlamlıysa)
            # ────────────────────────────────────────────────
            nemenyi_matrix = None
            if p < 0.05:
                report_lines.append("=" * 80)
                report_lines.append("🔍 NEMENYI POST-HOC TEST (Pairwise comparisons)")
                report_lines.append("=" * 80)
                report_lines.append("Performed only if Friedman p < 0.05")
                report_lines.append("")
                
                nemenyi = sp.posthoc_nemenyi_friedman(data_matrix)
                nemenyi.index = model_names
                nemenyi.columns = model_names
                
                # Matrisi rapor için güzel formatla
                report_lines.append("Nemenyi p-values (symmetric matrix):")
                report_lines.append(nemenyi.round(6).to_string())
                report_lines.append("")
                
                # Anlamlı çiftleri listele
                significant_pairs = []
                for i in range(len(model_names)):
                    for j in range(i+1, len(model_names)):
                        p_val = nemenyi.iloc[i, j]
                        if p_val < 0.05:
                            sig_marker = "***" if p_val < 0.01 else "**"
                            significant_pairs.append(
                                f"{model_names[i]} vs {model_names[j]}: p={p_val:.4f}{sig_marker}"
                            )
                
                if significant_pairs:
                    report_lines.append("Significant pairwise differences (p < 0.05):")
                    for pair in significant_pairs:
                        report_lines.append(f"  • {pair}")
                else:
                    report_lines.append("No significant pairwise differences found (all p ≥ 0.05)")
                report_lines.append("")
                
                nemenyi_matrix = nemenyi
            else:
                report_lines.append("Friedman testi anlamlı değil → Nemenyi post-hoc yapılmadı.")
                report_lines.append("")
            
            # ────────────────────────────────────────────────
            # 3. Özet Tablo
            # ────────────────────────────────────────────────
            report_lines.append("=" * 80)
            report_lines.append("📋 TEST SUMMARY & RECOMMENDATIONS")
            report_lines.append("=" * 80)
            report_lines.append("")
            
            summary_lines = [
                f"{'Test':<20} {'Purpose':<45} {'Result':<20}",
                "-" * 85,
                f"{'Friedman':<20} {'Overall difference among models':<45} {result:<20}",
            ]
            
            if p < 0.05:
                count_sig = len(significant_pairs) if 'significant_pairs' in locals() else 0
                summary_lines.append(
                    f"{'Nemenyi post-hoc':<20} {'Which pairs differ':<45} {count_sig} significant pairs"
                )
            else:
                summary_lines.append(
                    f"{'Nemenyi post-hoc':<20} {'Which pairs differ':<45} {'Not performed':<20}"
                )
            
            report_lines.extend(summary_lines)
            report_lines.append("")
            report_lines.append("💡 Practical Interpretation:")
            report_lines.append("  • Friedman → global ranking testi (tüm modeller aynı anda)")
            report_lines.append("  • Nemenyi → hangi modellerin birbirinden gerçekten farklı olduğunu söyler")
            report_lines.append("  • Eğer Nemenyi'de anlamlı çift yoksa → modeller pratikte çok yakın performans gösteriyor")
            report_lines.append("")
            
            # ────────────────────────────────────────────────
            # Raporu kaydet & yazdır
            # ────────────────────────────────────────────────
            report_text = "\n".join(report_lines)
            stat_path = "Friedman_Nemenyi_Statistical_Report.txt"
            
            with open(stat_path, "w", encoding="utf-8") as f:
                f.write(report_text)
            
            print(f"[OK] Friedman + Nemenyi statistical report saved: {stat_path}")
            print("\n" + report_text)
            
            # Eğer sınıfında safe_copy_to_drive varsa
            if hasattr(self, 'safe_copy_to_drive'):
                self.safe_copy_to_drive(stat_path, stat_path)
                
        except ImportError as ie:
            print(f"[ERROR] Missing library: {ie}")
            print("  → Make sure 'scikit-posthocs' is installed: pip install scikit-posthocs")
        except Exception as e:
            print(f"[ERROR] Statistical analysis failed: {e}")
            import traceback
            traceback.print_exc()

    def generate_holdout_reports(self, holdout_results, holdout_predictions, ablation_data, holdout_indices, labels):
        """Generate reports specifically for the holdout evaluation"""
        print("\n" + "="*80)
        print("=== GENERATING HOLDOUT REPORTS ===")
        print("="*80)
        
        # 1. Final Holdout Results
        df_holdout = pd.DataFrame(holdout_results)
        if not df_holdout.empty:
            self.save_report(df_holdout, "Final_Holdout_Test_Results.csv")
            
            # Confusion Matrices for Holdout
            for rec in holdout_results:
                model_name = rec['Model']
                try:
                    tn, fp, fn, tp = rec['TN'], rec['FP'], rec['FN'], rec['TP']
                    cm_arr = np.array([[tn, fp], [fn, tp]])
                    
                    plt.figure(figsize=(5, 4))
                    sns.heatmap(
                        cm_arr, annot=True, fmt=".0f", cbar=False,
                        xticklabels=['Pred: Normal', 'Pred: Phishing'],
                        yticklabels=['True: Normal', 'True: Phishing']
                    )
                    plt.title(f"Holdout CM - {model_name}")
                    plt.ylabel("True")
                    plt.xlabel("Pred")
                    plt.tight_layout()
                    
                    img_name = f"CM_Holdout_{model_name.replace(' ', '_')}.png"
                    plt.savefig(img_name, dpi=300)
                    plt.close()
                    self.safe_copy_to_drive(img_name, img_name)
                except Exception as e:
                    print(f"[ERROR] Failed to generate Holdout CM for {model_name}: {e}")
                    
        # 2. Holdout Ablation Study
        df_ablation = pd.DataFrame(ablation_data)
        if not df_ablation.empty:
            # Calculate accuracy drop from baseline
            baseline_accs = df_ablation[df_ablation['Scenario'] == 'BASELINE'].set_index('Model')['Accuracy']
            df_ablation['Accuracy_Drop'] = df_ablation.apply(
                lambda row: round(baseline_accs.get(row['Model'], row['Accuracy']) - row['Accuracy'], 6), axis=1
            )
            self.save_report(df_ablation, "Holdout_Ablation_Study.csv")
            
            # Summary
            abl_summary = df_ablation.groupby(['Scenario', 'Model'])[[
                'Accuracy','Precision','Recall','F1_Score','Sensitivity','Specificity',
                'FNR','FPR','Accuracy_Drop'
            ]].mean().round(6).reset_index()
            self.save_report(abl_summary, "Holdout_Ablation_Summary.csv")
            
            # Bar plot
            try:
                fig, ax = plt.subplots(figsize=(12, 6))
                df_bar = df_ablation.groupby(['Scenario', 'Model'])['Accuracy'].mean().unstack()
                df_bar.plot(kind='bar', ax=ax)
                ax.set_title("Holdout Ablation Study - Accuracy per Scenario & Model")
                ax.set_ylabel("Accuracy")
                ax.set_xlabel("Scenario")
                ax.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.xticks(rotation=30, ha='right')
                plt.tight_layout()
                abl_fig = "Holdout_Ablation_BarPlot.png"
                plt.savefig(abl_fig, dpi=300)
                plt.close()
                self.safe_copy_to_drive(abl_fig, abl_fig)
            except Exception as e:
                print(f"[ERROR] Holdout ablation bar plot failed: {e}")
                
        # 3. Holdout Statistical Analysis (McNemar-like)
        self._generate_holdout_statistical_report(holdout_predictions, holdout_indices, labels)
        
    def _generate_holdout_statistical_report(self, holdout_predictions, holdout_indices, labels):
        try:
            import scipy.stats as sp_stats
            
            model_names = list(holdout_predictions.keys())
            y_test_holdout = labels[holdout_indices]
            stat_lines = []
            
            stat_lines.append("="*80)
            stat_lines.append("HOLDOUT STATISTICAL ANALYSIS REPORT (McNemar-style)")
            stat_lines.append("="*80)
            
            for i, m1 in enumerate(model_names):
                for m2 in model_names[i+1:]:
                    correct_m1 = (holdout_predictions[m1] == y_test_holdout).astype(int)
                    correct_m2 = (holdout_predictions[m2] == y_test_holdout).astype(int)
                    
                    b = np.sum((correct_m1 == 1) & (correct_m2 == 0))
                    c = np.sum((correct_m1 == 0) & (correct_m2 == 1))
                    
                    n_discordant = b + c
                    if n_discordant > 0:
                        chi2 = (abs(b - c) - 1)**2 / (b + c)
                        p_mcnemar = 1 - sp_stats.chi2.cdf(chi2, df=1)
                        sig = "Significant" if p_mcnemar < 0.05 else "Not significant"
                        stat_lines.append(f"{m1} vs {m2}: b={b}, c={c}, chi2={chi2:.4f}, p={p_mcnemar:.6f} ({sig})")
                    else:
                        stat_lines.append(f"{m1} vs {m2}: identical predictions (no discordant pairs)")
                        
            report_text = "\n".join(stat_lines)
            stat_path = "Holdout_Statistical_Significance_Report.txt"
            
            with open(stat_path, "w", encoding="utf-8") as f:
                f.write(report_text)
            
            print(f"[OK] Holdout statistical report saved: {stat_path}")
            self.safe_copy_to_drive(stat_path, stat_path)
            
        except Exception as e:
            print(f"[ERROR] Holdout statistical analysis failed: {e}")