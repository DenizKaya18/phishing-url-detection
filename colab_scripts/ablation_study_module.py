# Düzeltilmiş CustomAblationStudy (yerine koyulacak tam sınıf)
import numpy as np
import pandas as pd
import time
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class CustomAblationStudy:
    """
    Özel ablation study - 5 senaryolu feature çıkarma analizi.

    Senaryolar:
    1. REMOVE BOW ONLY
    2. REMOVE SEG_BOW ONLY
    3. REMOVE NGRAMS + GRAMS4 (birlikte)
    4. REMOVE TLD ONLY
    5. REMOVE RATIO-BASED FEATURES (URL Length Ratio, Path Length Ratio, Domain Length Ratio)
    """

    def __init__(self, classifier, X_url_test, y_test, X_num_test_original):
        """
        Args:
            classifier: Eğitilmiş ensemble classifier (tokenizer ve scaler saklı olmalı)
            X_url_test: Test URLs (list-like)
            y_test: Test labels (np.array veya list)
            X_num_test_original: Original unscaled 20 numerical features (np.ndarray)
        """
        self.classifier = classifier
        self.X_url_test = X_url_test
        self.y_test = np.array(y_test)
        self.X_num_test_original = np.array(X_num_test_original, dtype=float)

        # Results
        self.baseline_metrics = None
        self.scenario_results = []
        self.comparison_df = None

        # Feature index mapping (kendi extractor sıralamana göre doğrula)
        # 0: URL Length
        # 1: Special Characters Count
        # 2: URL Length Ratio
        # 3: TLD Weight
        # 4-12: Security Flags ...
        # 13: Bag_of_Words_Count
        # 14: Weighted BoW (SUM Wi)
        # 15: Weighted Segmented BoW (SUM)
        # 16: Weighted 3-grams (SUM)
        # 17: Weighted 4-grams (SUM)
        # 18: Path ratio
        # 19: Domain ratio

        # Senaryolar tanımı
        self.scenarios = {
            'Scenario_1_Remove_BoW': {
                'name': 'Scenario 1: Remove BoW Features Only',
                'description': 'BoW (Bag of Words) features çıkarıldı',
                'indices_to_remove': [14],  # Only Weighted BoW
                'order': 1
            },
            'Scenario_2_Remove_SegBoW': {
                'name': 'Scenario 2: Remove Segmented BoW Only',
                'description': 'Segmented BoW features çıkarıldı',
                'indices_to_remove': [15],  # Only Seg BoW
                'order': 2
            },
            'Scenario_3_Remove_NGrams': {
                'name': 'Scenario 3: Remove N-grams + 4-grams',
                'description': '3-grams ve 4-grams features birlikte çıkarıldı',
                'indices_to_remove': [16, 17],  # Both n-grams
                'order': 3
            },
            'Scenario_4_Remove_TLD': {
                'name': 'Scenario 4: Remove TLD Weight Only',
                'description': 'TLD Weight features çıkarıldı',
                'indices_to_remove': [3],  # Only TLD weight
                'order': 4
            },
            'Scenario_5_Remove_RatioBased': {
                'name': 'Scenario 5: Remove Ratio-Based Features',
                'description': 'Ratio-based features çıkarıldı (URL Length Ratio, Path Length Ratio, Domain Length Ratio)',
                'indices_to_remove': [2, 18, 19],  # URL_Length_Ratio, path_ratio, domain_ratio
                'order': 5
            }
        }

        print("✓ Custom Ablation Study Başlatıldı")
        print(f"✓ {len(self.scenarios)} senaryo tanımlandı")

    def _remove_features(self, indices_to_remove: List[int]) -> np.ndarray:
        """
        Belirtilen indices'teki features'ları 0'la set et.
        """
        X_modified = self.X_num_test_original.copy()
        # Validate indices
        n_cols = X_modified.shape[1]
        for idx in indices_to_remove:
            if idx < 0 or idx >= n_cols:
                raise IndexError(f"Index {idx} is out of range for features shape {X_modified.shape}")
        X_modified[:, indices_to_remove] = 0.0
        return X_modified

    def _evaluate_configuration(self, X_num_test_config: np.ndarray,
                               scenario_name: str, scenario_desc: str) -> Dict:
        """
        Güvenli evaluation: test üzerinde scaler.fit yapılmaz.
        """
        print(f"\n⏳ Evaluating: {scenario_name}...")
        print(f"   Description: {scenario_desc}")
        start_time = time.time()

        try:
            X_num_test_config = np.array(X_num_test_config, dtype=float)
            # --- SANITY: check removed columns are zeroed (first 5 rows sample) ---
            zeroed_cols = [c for c in range(X_num_test_config.shape[1]) if np.allclose(X_num_test_config[:, c], 0.0)]
            if zeroed_cols:
                print(f"   ✓ Zeroed columns sample (indices): {zeroed_cols[:10]}")

            # --- SCALING: use fitted scaler from classifier ---
            if not hasattr(self.classifier, 'scaler') or self.classifier.scaler is None:
                raise RuntimeError("Classifier has no fitted scaler (classifier.scaler). Fit scaler on training and attach as classifier.scaler before running ablation.")
            scaler = self.classifier.scaler

            # Check feature dimension match
            expected_n = getattr(scaler, 'n_features_in_', None)
            if expected_n is not None and expected_n != X_num_test_config.shape[1]:
                raise RuntimeError(f"Feature dimension mismatch: scaler expects {expected_n} features but test has {X_num_test_config.shape[1]}")

            X_num_test_scaled = scaler.transform(X_num_test_config)

            # --- TOKENIZE URLs using classifier.tokenizer (already fit on train) ---
            if not hasattr(self.classifier, 'tokenizer') or self.classifier.tokenizer is None:
                raise RuntimeError("Classifier has no tokenizer (classifier.tokenizer). Fit tokenizer on training and attach as classifier.tokenizer before running ablation.")
            seq_test = self.classifier.tokenizer.texts_to_sequences(self.X_url_test)
            X_url_test_pad = pad_sequences(seq_test, maxlen=self.classifier.max_len,
                                          padding="post", truncating="post")

            # Ensemble prediction - dynamic number of models
            ensemble_pred_proba = []
            n_models = len(getattr(self.classifier, 'models', []))
            if n_models == 0:
                raise RuntimeError("Classifier has no models (classifier.models is empty). Train or load models before running ablation.")

            for model_idx, model in enumerate(self.classifier.models):
                pred_proba = model.predict(
                    {"url_input": X_url_test_pad, "num_input": X_num_test_scaled},
                    verbose=0, batch_size=512
                ).flatten()
                ensemble_pred_proba.append(pred_proba)
                print(f"   Model {model_idx+1}/{n_models} prediction completed", end='\r')

            print(f"   Model {n_models}/{n_models} prediction completed")

            # Soft voting
            ensemble_pred_proba = np.mean(ensemble_pred_proba, axis=0)
            ensemble_pred = (ensemble_pred_proba > 0.5).astype(int)

            # Calculate metrics (assumes classifier has calculate_metrics method)
            metrics = self.classifier.calculate_metrics(self.y_test, ensemble_pred)

            elapsed = time.time() - start_time
            metrics['eval_time'] = elapsed

            print(f"   ✓ Accuracy: {metrics['accuracy']:.4f}")
            print(f"   ✓ Precision: {metrics['precision']:.4f}")
            print(f"   ✓ Recall: {metrics['recall']:.4f}")
            print(f"   ✓ F1-Score: {metrics['f1_score']:.4f}")
            print(f"   ✓ Time: {elapsed:.2f}s")

            return metrics

        except Exception as e:
            print(f"   ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def run_custom_ablation(self) -> pd.DataFrame:
        """
        5 senaryolu custom ablation study'yi çalıştır.
        """
        print("\n" + "="*100)
        print("🔬 CUSTOM ABLATION STUDY - 5 SCENARIOS")
        print("="*100)

        results = []

        # ==================== BASELINE ====================
        print("\n" + "="*100)
        print("[SCENARIO 0 - BASELINE] Tüm features ile başlangıç...")
        print("="*100)

        X_num_baseline = self.X_num_test_original.copy()
        baseline = self._evaluate_configuration(X_num_baseline, "BASELINE (All Features)",
                                               "Tüm features kullanılıyor")

        if baseline is None:
            raise RuntimeError("Baseline evaluation failed. Check classifier.scaler/tokenizer/models and X_num_test_original.")

        self.baseline_metrics = baseline

        results.append({
            'Scenario': 'BASELINE (All Features)',
            'Removed_Features': 'None',
            'Accuracy': baseline['accuracy'],
            'Precision': baseline['precision'],
            'Recall': baseline['recall'],
            'F1_Score': baseline['f1_score'],
            'Sensitivity': baseline['sensitivity'],
            'Specificity': baseline['specificity'],
            'Confusion_Matrix': baseline['confusion_matrix'],
            'Evaluation_Time_s': baseline['eval_time'],
            'Accuracy_Drop': 0.0,
            'F1_Drop': 0.0
        })

        # ==================== 5 SENARYOLAR ====================
        for scenario_key in sorted(self.scenarios.keys(),
                                  key=lambda x: self.scenarios[x]['order']):
            scenario = self.scenarios[scenario_key]

            print("\n" + "="*100)
            print(f"[{scenario['name']}]")
            print("="*100)

            # Remove features
            X_num_modified = self._remove_features(scenario['indices_to_remove'])

            # Evaluate
            metrics = self._evaluate_configuration(
                X_num_modified,
                scenario['name'],
                scenario['description']
            )

            if metrics is not None:
                accuracy_drop = baseline['accuracy'] - metrics['accuracy']
                f1_drop = baseline['f1_score'] - metrics['f1_score']

                results.append({
                    'Scenario': scenario['name'],
                    'Removed_Features': ', '.join(map(str, scenario['indices_to_remove'])),
                    'Accuracy': metrics['accuracy'],
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall'],
                    'F1_Score': metrics['f1_score'],
                    'Sensitivity': metrics['sensitivity'],
                    'Specificity': metrics['specificity'],
                    'Confusion_Matrix': metrics['confusion_matrix'],
                    'Evaluation_Time_s': metrics['eval_time'],
                    'Accuracy_Drop': accuracy_drop,
                    'F1_Drop': f1_drop
                })
            else:
                print(f"   ⚠️ Skipping scenario {scenario['name']} due to evaluation error.")

        # Convert to DataFrame
        self.comparison_df = pd.DataFrame(results)

        return self.comparison_df

    def print_detailed_report(self):
        """
        Detaylı ablation study raporu.
        """
        print("\n" + "="*120)
        print("📊 CUSTOM ABLATION STUDY - DETAILED RESULTS")
        print("="*120)

        # Baseline
        print("\n[BASELINE - ALL FEATURES]")
        print("-"*80)
        if self.baseline_metrics:
            print(f"Accuracy:   {self.baseline_metrics['accuracy']:.4f}")
            print(f"Precision:  {self.baseline_metrics['precision']:.4f}")
            print(f"Recall:     {self.baseline_metrics['recall']:.4f}")
            print(f"F1-Score:   {self.baseline_metrics['f1_score']:.4f}")
            print(f"Sensitivity:{self.baseline_metrics['sensitivity']:.4f}")
            print(f"Specificity:{self.baseline_metrics['specificity']:.4f}")
            print(f"Time:       {self.baseline_metrics['eval_time']:.2f}s")

        # Scenario results
        print("\n[SCENARIO COMPARISON]")
        print("-"*120)
        print(f"{'Scenario':<35} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Acc Drop':<12}")
        print("-"*120)

        if self.comparison_df is None or self.comparison_df.empty:
            print("No scenario results to show.")
            return

        for _, row in self.comparison_df.iterrows():
            if row['Scenario'] != 'BASELINE (All Features)':
                print(f"{row['Scenario']:<35} {row['Accuracy']:<12.4f} {row['Precision']:<12.4f} "
                      f"{row['Recall']:<12.4f} {row['F1_Score']:<12.4f} {row['Accuracy_Drop']:<12.4f}")

        # Feature importance ranking
        print("\n[FEATURE IMPORTANCE RANKING - By Accuracy Drop]")
        print("-"*80)
        print(f"{'Rank':<6} {'Scenario':<35} {'Accuracy Drop':<15} {'F1 Drop':<15}")
        print("-"*80)

        scenario_df = self.comparison_df[self.comparison_df['Scenario'] != 'BASELINE (All Features)'].copy()
        scenario_df = scenario_df.sort_values('Accuracy_Drop', ascending=False)

        for idx, (_, row) in enumerate(scenario_df.iterrows(), 1):
            print(f"{idx:<6} {row['Scenario']:<35} {row['Accuracy_Drop']:<15.4f} {row['F1_Drop']:<15.4f}")

        # Detailed comparison table
        print("\n[DETAILED METRICS COMPARISON]")
        print("-"*140)
        print(f"{'Scenario':<35} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Sensitivity':<12} {'Specificity':<12} {'Time(s)':<10}")
        print("-"*140)

        for _, row in self.comparison_df.iterrows():
            scenario_short = row['Scenario'].replace('Scenario ', 'S').replace(': ', ': ')[:35]
            print(f"{scenario_short:<35} {row['Accuracy']:<12.4f} {row['Precision']:<12.4f} "
                  f"{row['Recall']:<12.4f} {row['F1_Score']:<12.4f} {row['Sensitivity']:<12.4f} "
                  f"{row['Specificity']:<12.4f} {row['Evaluation_Time_s']:<10.2f}")

        print("\n" + "="*120)

    def print_confusion_matrices(self):
        """
        Confusion matrixleri görüntüle.
        """
        print("\n" + "="*100)
        print("🔍 CONFUSION MATRICES FOR ALL SCENARIOS")
        print("="*100)

        if self.comparison_df is None:
            print("No comparison data available.")
            return

        for _, row in self.comparison_df.iterrows():
            scenario = row['Scenario']
            cm = row['Confusion_Matrix']

            print(f"\n[{scenario}]")
            print("-"*60)

            if isinstance(cm, (np.ndarray, list, tuple)):
                cm_arr = np.array(cm)
                if cm_arr.shape == (2, 2):
                    tn, fp, fn, tp = cm_arr.ravel()
                    print(pd.DataFrame([[int(tn), int(fp)], [int(fn), int(tp)]],
                                       index=['Actual:Benign','Actual:Malicious'],
                                       columns=['Pred:Benign','Pred:Malicious']))
                    print(f"TN={int(tn)}, FP={int(fp)}, FN={int(fn)}, TP={int(tp)}")
                else:
                    print(cm_arr)
            else:
                print(cm)

        print("\n" + "="*100)

    def export_results(self, filename="custom_ablation_results.csv"):
        """
        Sonuçları CSV'ye kaydet.
        """
        if self.comparison_df is not None:
            export_df = self.comparison_df.copy()
            export_df['Confusion_Matrix'] = export_df['Confusion_Matrix'].astype(str)

            export_df.to_csv(filename, index=False)
            print(f"\n✓ Results exported to: {filename}")

            # Summary statistics
            summary_file = filename.replace('.csv', '_summary.txt')
            with open(summary_file, 'w') as f:
                f.write("="*100 + "\n")
                f.write("CUSTOM ABLATION STUDY - SUMMARY\n")
                f.write("="*100 + "\n\n")

                f.write("BASELINE METRICS\n")
                f.write("-"*60 + "\n")
                if self.baseline_metrics:
                    f.write(f"Accuracy:   {self.baseline_metrics['accuracy']:.4f}\n")
                    f.write(f"Precision:  {self.baseline_metrics['precision']:.4f}\n")
                    f.write(f"Recall:     {self.baseline_metrics['recall']:.4f}\n")
                    f.write(f"F1-Score:   {self.baseline_metrics['f1_score']:.4f}\n")

                f.write("\n" + "="*100 + "\n")
                f.write("SCENARIO RESULTS\n")
                f.write("="*100 + "\n\n")

                if self.comparison_df is not None and not self.comparison_df.empty:
                    scenario_df = self.comparison_df[self.comparison_df['Scenario'] != 'BASELINE (All Features)'].copy()
                    scenario_df = scenario_df.sort_values('Accuracy_Drop', ascending=False)

                    f.write(f"{'Scenario':<50} {'Accuracy Drop':<20} {'F1 Drop':<20}\n")
                    f.write("-"*90 + "\n")

                    for _, row in scenario_df.iterrows():
                        f.write(f"{row['Scenario']:<50} {row['Accuracy_Drop']:<20.4f} {row['F1_Drop']:<20.4f}\n")

            print(f"✓ Summary exported to: {summary_file}")

    def create_visualizations(self, save_dir="custom_ablation_plots/"):
        """
        Visualizasyonlar oluştur.
        """
        import os
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if self.comparison_df is None or self.comparison_df.empty:
            print("No comparison data to plot.")
            return

        scenario_df = self.comparison_df[self.comparison_df['Scenario'] != 'BASELINE (All Features)'].copy()
        scenario_df = scenario_df.sort_values('Accuracy_Drop', ascending=False)

        # PLOT 1: Accuracy Drop
        plt.figure(figsize=(14, 6))
        colors = ['#d62728' if x > 0 else '#2ca02c' for x in scenario_df['Accuracy_Drop']]
        bars = plt.bar(range(len(scenario_df)), scenario_df['Accuracy_Drop'], color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

        for i, (bar, val) in enumerate(zip(bars, scenario_df['Accuracy_Drop'])):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
                    f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        plt.xlabel('Scenario', fontsize=12, fontweight='bold')
        plt.ylabel('Accuracy Drop', fontsize=12, fontweight='bold')
        plt.title('Feature Contribution Analysis - Accuracy Drop When Removing Features',
                 fontsize=14, fontweight='bold')
        plt.xticks(range(len(scenario_df)),
                   [s.replace('Scenario ', 'S').replace(': ', ':\n')[:30] for s in scenario_df['Scenario']],
                   rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"{save_dir}01_accuracy_drop_comparison.png", dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_dir}01_accuracy_drop_comparison.png")
        plt.close()

        # PLOT 2: Metrics Comparison
        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1_Score']

        fig, ax = plt.subplots(figsize=(14, 7))
        x = np.arange(len(scenario_df))
        width = 0.2

        for i, metric in enumerate(metrics_to_plot):
            offset = (i - 1.5) * width
            values = scenario_df[metric].values
            ax.bar(x + offset, values, width, label=metric, alpha=0.8, edgecolor='black', linewidth=0.5)

        baseline_acc = self.baseline_metrics['accuracy']
        ax.axhline(y=baseline_acc, color='red', linestyle='--', linewidth=2, label=f'Baseline ({baseline_acc:.4f})')

        ax.set_xlabel('Scenario', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Performance Metrics Comparison Across Scenarios', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace('Scenario ', 'S').replace(': ', ':\n')[:30] for s in scenario_df['Scenario']],
                           rotation=45, ha='right')
        ax.legend(loc='lower left', fontsize=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(f"{save_dir}02_metrics_comparison.png", dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_dir}02_metrics_comparison.png")
        plt.close()

        # Remaining plots (same as before)...
        # (I kept rest of plotting code in original but you can reuse from your previous implementation)
        print(f"\n✓ All visualizations saved to: {save_dir}")
