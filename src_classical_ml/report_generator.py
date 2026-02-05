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
        """Generate comprehensive statistical significance report"""
        if df_base.empty:
            print("[WARNING] No baseline data for statistical analysis")
            return
        
        try:
            # Import gerekli kütüphaneler
            import numpy as np
            from scipy.stats import ttest_rel, wilcoxon, f_oneway, chi2_contingency
            from scipy.stats import chi2
            
            # Collect results per model
            results_per_model = {}
            for model in df_base['Model'].unique():
                model_data = df_base[df_base['Model'] == model]
                accuracies = model_data['Accuracy'].values.tolist()
                
                # Try to get predictions if available
                y_true = model_data['y_true'].values if 'y_true' in model_data.columns else None
                y_pred = model_data['y_pred'].values if 'y_pred' in model_data.columns else None
                
                results_per_model[model] = {
                    'fold_accuracies': accuracies,
                    'y_true': y_true,
                    'y_pred': y_pred
                }
            
            model_names = list(results_per_model.keys())
            if not model_names:
                print("[WARNING] No models found for statistical analysis")
                return
                
            n_folds = len(results_per_model[model_names[0]]['fold_accuracies'])
            
            report_lines = []
            report_lines.append("=" * 80)
            report_lines.append("📊 COMPREHENSIVE STATISTICAL SIGNIFICANCE REPORT")
            report_lines.append("Classical ML Models - Cross-Validated")
            report_lines.append("=" * 80)
            report_lines.append("")
            
            # ==================== McNEMAR TEST ====================
            if all(results_per_model[m]['y_true'] is not None and 
                results_per_model[m]['y_pred'] is not None for m in model_names):
                
                report_lines.append("=" * 80)
                report_lines.append("🔬 McNEMAR TEST: PAIRWISE MODEL COMPARISONS")
                report_lines.append("=" * 80)
                report_lines.append("Testing if models have significantly different error patterns")
                report_lines.append("")
                
                y_true_all = np.array(results_per_model[model_names[0]]['y_true'])
                
                header = f"{'Model A':<20} {'Model B':<20} {'Acc A':<10} {'Acc B':<10} {'χ²':<10} {'p-value':<10} {'Result':<15}"
                report_lines.append(header)
                report_lines.append("-" * 95)
                
                for i, model_a in enumerate(model_names):
                    for model_b in model_names[i+1:]:
                        pred_a = np.array(results_per_model[model_a]['y_pred'])
                        pred_b = np.array(results_per_model[model_b]['y_pred'])
                        
                        correct_a = (pred_a == y_true_all).astype(int)
                        correct_b = (pred_b == y_true_all).astype(int)
                        
                        # McNemar contingency
                        a01 = np.sum((correct_a == 0) & (correct_b == 1))
                        a10 = np.sum((correct_a == 1) & (correct_b == 0))
                        
                        if (a01 + a10) == 0:
                            chi2_stat = 0
                            p_value = 1.0
                        else:
                            chi2_stat = ((abs(a10 - a01) - 1) ** 2) / (a10 + a01)  # Continuity correction
                            p_value = 1 - chi2.cdf(chi2_stat, df=1)
                        
                        acc_a = np.mean(correct_a)
                        acc_b = np.mean(correct_b)
                        
                        sig_marker = "***" if p_value < 0.01 else "**" if p_value < 0.05 else ""
                        result = f"Significant{sig_marker}" if p_value < 0.05 else "Not significant"
                        
                        line = f"{model_a:<20} {model_b:<20} {acc_a:<10.4f} {acc_b:<10.4f} {chi2_stat:<10.4f} {p_value:<10.6f} {result:<15}"
                        report_lines.append(line)
                
                report_lines.append("-" * 95)
                report_lines.append("Legend: *** p<0.01 (highly significant), ** p<0.05 (significant)")
                report_lines.append("")
            
            # ==================== PAIRED T-TEST ====================
            report_lines.append("=" * 80)
            report_lines.append("📈 PAIRED T-TEST: ALL MODELS COMPARISON (ACCURACY)")
            report_lines.append("=" * 80)
            report_lines.append(f"Comparing accuracy across {n_folds} folds")
            report_lines.append("")
            
            # Summary statistics
            report_lines.append(f"{'Model':<20} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
            report_lines.append("-" * 60)
            
            for model_name in model_names:
                accs = np.array(results_per_model[model_name]['fold_accuracies'])
                line = f"{model_name:<20} {np.mean(accs):<10.4f} {np.std(accs):<10.4f} {np.min(accs):<10.4f} {np.max(accs):<10.4f}"
                report_lines.append(line)
            
            report_lines.append("-" * 60)
            report_lines.append("")
            
            # Pairwise comparisons
            header = f"{'Model A':<20} {'Model B':<20} {'t-stat':<10} {'p-value':<10} {'Result':<15}"
            report_lines.append(header)
            report_lines.append("-" * 75)
            
            for i, m1 in enumerate(model_names):
                for m2 in model_names[i+1:]:
                    a = np.array(results_per_model[m1]['fold_accuracies'])
                    b = np.array(results_per_model[m2]['fold_accuracies'])
                    
                    if len(a) == len(b) and len(a) > 1:
                        t, p = ttest_rel(a, b)
                        sig_marker = "***" if p < 0.01 else "**" if p < 0.05 else ""
                        res = f"Significant{sig_marker}" if p < 0.05 else "Not significant"
                        line = f"{m1:<20} {m2:<20} {t:<10.4f} {p:<10.6f} {res:<15}"
                        report_lines.append(line)
            
            report_lines.append("-" * 75)
            report_lines.append("Legend: *** p<0.01, ** p<0.05")
            report_lines.append("")
            report_lines.append("💡 INTERPRETATION:")
            report_lines.append("  • Paired t-test compares average performance across folds")
            report_lines.append("  • H0: Both models perform equally on average")
            report_lines.append("  • If p<0.05: Significant performance difference")
            report_lines.append("")
            
            # ==================== WILCOXON TEST ====================
            report_lines.append("=" * 80)
            report_lines.append("📊 WILCOXON TEST: ALL MODELS COMPARISON (ACCURACY)")
            report_lines.append("=" * 80)
            report_lines.append(f"Non-parametric comparison across {n_folds} folds")
            report_lines.append("")
            
            header = f"{'Model A':<20} {'Model B':<20} {'Median A':<12} {'Median B':<12} {'W-stat':<10} {'p-value':<10} {'Result':<15}"
            report_lines.append(header)
            report_lines.append("-" * 85)
            
            for i, m1 in enumerate(model_names):
                for m2 in model_names[i+1:]:
                    a = np.array(results_per_model[m1]['fold_accuracies'])
                    b = np.array(results_per_model[m2]['fold_accuracies'])
                    
                    if len(a) == len(b) and len(a) > 0:
                        try:
                            w, p = wilcoxon(a, b)
                            median_a = np.median(a)
                            median_b = np.median(b)
                            sig_marker = "***" if p < 0.01 else "**" if p < 0.05 else ""
                            res = f"Significant{sig_marker}" if p < 0.05 else "Not significant"
                            line = f"{m1:<20} {m2:<20} {median_a:<12.4f} {median_b:<12.4f} {w:<10.4f} {p:<10.6f} {res:<15}"
                            report_lines.append(line)
                        except Exception as e:
                            line = f"{m1:<20} {m2:<20} Wilcoxon failed ({str(e)[:30]})"
                            report_lines.append(line)
            
            report_lines.append("-" * 85)
            report_lines.append("💡 INTERPRETATION:")
            report_lines.append("  • Non-parametric alternative to paired t-test")
            report_lines.append("  • More robust to outliers and non-normal distributions")
            report_lines.append("")
            
            # ==================== ANOVA ====================
            report_lines.append("=" * 80)
            report_lines.append("📉 ONE-WAY ANOVA: ALL MODELS COMPARISON (ACCURACY)")
            report_lines.append("=" * 80)
            report_lines.append(f"Testing if performance differs among all {len(model_names)} models")
            report_lines.append("")
            
            all_data = [np.array(results_per_model[m]['fold_accuracies'])
                    for m in model_names if len(results_per_model[m]['fold_accuracies']) > 0]
            
            if len(all_data) > 1:
                f_stat, p = f_oneway(*all_data)
                
                report_lines.append(f"F-statistic: {f_stat:.4f}")
                report_lines.append(f"p-value: {p:.6f}")
                
                sig_marker = "***" if p < 0.01 else "**" if p < 0.05 else ""
                significant = f"Yes{sig_marker}" if p < 0.05 else "No"
                report_lines.append(f"Significant difference (α=0.05): {significant}")
                report_lines.append("")
                report_lines.append("💡 INTERPRETATION:")
                report_lines.append("  • H0: All models perform equally well on average")
                report_lines.append("  • H1: At least one model performs significantly differently")
                report_lines.append("  • If p < 0.05: Reject H0 (significant differences exist)")
                report_lines.append("")
            
            # ==================== COHEN'S D ====================
            report_lines.append("=" * 80)
            report_lines.append("📏 EFFECT SIZES: COHEN'S D (Pairwise Comparisons)")
            report_lines.append("=" * 80)
            report_lines.append("")
            
            header = f"{'Model A':<20} {'Model B':<20} {'Cohens d':<12} {'|Effect|':<12} {'Interpretation':<25}"
            report_lines.append(header)
            report_lines.append("-" * 95)
            
            for i, m1 in enumerate(model_names):
                for m2 in model_names[i+1:]:
                    a = np.array(results_per_model[m1]['fold_accuracies'])
                    b = np.array(results_per_model[m2]['fold_accuracies'])
                    
                    if len(a) == len(b) and len(a) > 1:
                        mean_a, mean_b = a.mean(), b.mean()
                        std_a, std_b = a.std(ddof=1), b.std(ddof=1)
                        pooled_std = np.sqrt(
                            ((len(a)-1)*std_a**2 + (len(b)-1)*std_b**2) / (len(a)+len(b)-2)
                        )
                        d = (mean_a - mean_b) / pooled_std if pooled_std > 0 else 0
                        
                        abs_d = abs(d)
                        if abs_d < 0.2:
                            effect = "Negligible"
                        elif abs_d < 0.5:
                            effect = "Small"
                        elif abs_d < 0.8:
                            effect = "Medium"
                        else:
                            effect = "Large"
                        
                        direction = "A better" if d > 0 else "B better" if d < 0 else "Equal"
                        
                        line = f"{m1:<20} {m2:<20} {d:<12.4f} {abs_d:<12.4f} {effect} ({direction})"
                        report_lines.append(line)
            
            report_lines.append("-" * 95)
            report_lines.append("💡 COHEN'S D INTERPRETATION:")
            report_lines.append("  • 0.0 - 0.2: Negligible effect")
            report_lines.append("  • 0.2 - 0.5: Small effect")
            report_lines.append("  • 0.5 - 0.8: Medium effect")
            report_lines.append("  • 0.8+:      Large effect")
            report_lines.append("")
            
            # ==================== SUMMARY ====================
            report_lines.append("=" * 80)
            report_lines.append("📋 TEST SUMMARY")
            report_lines.append("=" * 80)
            report_lines.append("")
            
            summary_header = f"{'Test':<20} {'Purpose':<40} {'Models Tested':<15}"
            report_lines.append(summary_header)
            report_lines.append("-" * 75)
            report_lines.append(f"{'McNemar':<20} {'Error pattern differences':<40} {'All pairs':<15}")
            report_lines.append(f"{'Paired t-test':<20} {'Mean accuracy differences':<40} {'All pairs':<15}")
            report_lines.append(f"{'Wilcoxon':<20} {'Non-parametric comparison':<40} {'All pairs':<15}")
            report_lines.append(f"{'One-way ANOVA':<20} {'All models vs each other':<40} {'All models':<15}")
            report_lines.append(f"{'Cohens d':<20} {'Effect size magnitude':<40} {'All pairs':<15}")
            report_lines.append("")
            
            report_lines.append("=" * 80)
            report_lines.append("✅ Statistical Analysis Complete")
            report_lines.append("=" * 80)
            
            # Save report
            stat_path = "Statistical_Significance_Report.txt"
            with open(stat_path, "w", encoding="utf-8") as f:
                f.write("\n".join(report_lines))
            print(f"[OK] Statistical report saved: {stat_path}")
            
            # Print to console
            print("\n".join(report_lines))
            
            # Eğer safe_copy_to_drive method'u varsa
            if hasattr(self, 'safe_copy_to_drive'):
                self.safe_copy_to_drive(stat_path, stat_path)
            
        except Exception as e:
            print(f"[ERROR] Statistical analysis failed: {e}")
            import traceback
            traceback.print_exc()