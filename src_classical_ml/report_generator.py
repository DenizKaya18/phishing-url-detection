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
        """Generate statistical significance report"""
        if df_base.empty:
            print("[WARNING] No baseline data for statistical analysis")
            return
        
        try:
            from scipy.stats import ttest_rel, wilcoxon, f_oneway
            
            # Collect fold accuracies per model
            results_per_model = {}
            for model in df_base['Model'].unique():
                accuracies = df_base[df_base['Model'] == model]['Accuracy'].values.tolist()
                results_per_model[model] = {'fold_accuracies': accuracies}
            
            model_names = list(results_per_model.keys())
            report_lines = []
            
            # Paired t-test
            report_lines.append("\n--- Paired T-Test ---")
            for i, m1 in enumerate(model_names):
                for m2 in model_names[i+1:]:
                    a = np.array(results_per_model[m1]['fold_accuracies'])
                    b = np.array(results_per_model[m2]['fold_accuracies'])
                    
                    if len(a) == len(b) and len(a) > 1:
                        t, p = ttest_rel(a, b)
                        res = "Significant" if p < 0.05 else "Not significant"
                        report_lines.append(f"{m1} vs {m2}: t={t:.4f}, p={p:.6f} ({res})")
            
            # Wilcoxon test
            report_lines.append("\n--- Wilcoxon Test ---")
            for i, m1 in enumerate(model_names):
                for m2 in model_names[i+1:]:
                    a = np.array(results_per_model[m1]['fold_accuracies'])
                    b = np.array(results_per_model[m2]['fold_accuracies'])
                    
                    if len(a) == len(b) and len(a) > 0:
                        try:
                            w, p = wilcoxon(a, b)
                            res = "Significant" if p < 0.05 else "Not significant"
                            report_lines.append(f"{m1} vs {m2}: w={w:.4f}, p={p:.6f} ({res})")
                        except Exception as e:
                            report_lines.append(f"{m1} vs {m2}: wilcoxon failed ({e})")
            
            # ANOVA
            report_lines.append("\n--- ANOVA ---")
            all_data = [np.array(results_per_model[m]['fold_accuracies']) 
                       for m in model_names if len(results_per_model[m]['fold_accuracies']) > 0]
            if len(all_data) > 1:
                f, p = f_oneway(*all_data)
                report_lines.append(f"F={f:.4f}, p={p:.6f}")
            
            # Cohen's d
            report_lines.append("\n--- Cohen's d ---")
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
                        report_lines.append(f"{m1} vs {m2}: d={d:.4f}")
            
            # Save report
            stat_path = "Statistical_Significance_Report.txt"
            with open(stat_path, "w", encoding="utf-8") as f:
                f.write("\n".join(report_lines))
            print(f"[OK] Statistical report saved: {stat_path}")
            
            self.safe_copy_to_drive(stat_path, stat_path)
            
        except Exception as e:
            print(f"[ERROR] Statistical analysis failed: {e}")
