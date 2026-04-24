# -*- coding: utf-8 -*-
"""
Model Training and Cross-Validation Module
"""

import os
import time
import json
import shutil
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import RandomOverSampler
from tqdm import tqdm

from evaluator import ModelEvaluator
from checkpoint import CheckpointManager


class ModelTrainer:
    """Handles model training with cross-validation"""
    
    def __init__(self, config, feature_builder, models):
        self.config = config
        self.feature_builder = feature_builder
        self.models = models
        self.evaluator = ModelEvaluator(config)
        self.checkpoint = CheckpointManager(config)
        
        # Load existing results
        self.full_report_data = []
        self.completed_folds = set()
        self.existing_keys = set()
        
        self._load_checkpoint()
    
    def _load_checkpoint(self):
        """Load checkpoint data"""
        self.full_report_data, self.completed_folds, self.existing_keys = \
            self.checkpoint.load_checkpoint()
        
        print(f"[CHECKPOINT] Loaded: {len(self.full_report_data)} records, "
              f"{len(self.completed_folds)} completed folds")
    
    def _upsert_record(self, record):
        """Insert or update record in results"""
        key = (record.get('Scenario'), int(record.get('Fold')), record.get('Model'))
        
        if key in self.existing_keys:
            # Replace existing
            for idx, old in enumerate(self.full_report_data):
                try:
                    old_key = (old.get('Scenario'), int(old.get('Fold')), old.get('Model'))
                except Exception:
                    old_key = None
                if old_key == key:
                    self.full_report_data[idx] = record
                    return
            self.full_report_data.append(record)
        else:
            self.full_report_data.append(record)
            self.existing_keys.add(key)
    
    def train_model(self, clf, X_train, y_train):
        """Train a model with class balancing"""
        try:
            # Try sample weights
            classes = np.unique(y_train)
            weights = compute_class_weight('balanced', classes=classes, y=y_train)
            sample_weights = np.array([weights[int(y)] for y in y_train])
            clf.fit(X_train, y_train, sample_weight=sample_weights)
            return clf, "sample_weight"
        except Exception:
            # Fall back to oversampling
            ros = RandomOverSampler(random_state=42)
            X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
            clf.fit(X_resampled, y_resampled)
            return clf, "resample"
    
    def run_cross_validation(self, X_all, y_all, labels):
        """Run cross-validation with all models"""
        cv = StratifiedKFold(
            n_splits=self.config.n_splits,
            shuffle=True,
            random_state=self.config.random_state
        )
        
        scenarios = {'BASELINE': []}  # Only baseline scenario
        total_steps_per_fold = len(scenarios) * len(self.models)
        
        for fold, (train_idx, test_idx) in enumerate(cv.split(np.arange(len(y_all)), y_all), start=1):
            # Skip completed folds
            if fold in self.completed_folds:
                print(f"\n=== Fold {fold} already completed (checkpoint). Skipping. ===")
                continue
            
            print(f"\n{'='*60}\n=== Fold {fold} Starting ===\n{'='*60}")
            
            # Build features for this fold
            self.feature_builder.store.clear()
            self.feature_builder.build_counters_from_train(
                train_idx, labels, threshold=self.config.feature_threshold
            )
            
            X_train_base = self.feature_builder.build_features(train_idx)
            X_test_base = self.feature_builder.build_features(test_idx)
            y_train_fold = y_all[train_idx]
            y_test_fold = y_all[test_idx]
            
            # Calculate imbalance ratio
            c0 = np.sum(y_train_fold == 0)
            c1 = np.sum(y_train_fold == 1)
            imb_ratio = c0 / c1 if c1 > 0 else 0
            
            print(f"-> Features ready. Train: {X_train_base.shape}, Test: {X_test_base.shape}")
            
            # Train all models
            with tqdm(total=total_steps_per_fold, desc=f"Fold {fold} progress", unit="step") as fold_pbar:
                for scen_name, remove_cols in scenarios.items():
                    print(f"   Scenario: {scen_name} (Fold {fold})")
                    
                    X_train = X_train_base
                    X_test = X_test_base
                    
                    for model_name, model in self.models.items():
                        model_desc = f"{scen_name} | {model_name}"
                        fold_pbar.set_description(f"Fold {fold}: {model_desc}")
                        fold_pbar.refresh()
                        
                        # Train model
                        clf = clone(model)
                        t0 = time.time()
                        clf, fit_mode = self.train_model(clf, X_train, y_train_fold)
                        train_time = time.time() - t0
                        
                        # Evaluate
                        metrics = self.evaluator.evaluate_model(
                            clf, X_test, y_test_fold, model_name
                        )
                        
                        # Create record
                        record = {
                            'Scenario': scen_name,
                            'Fold': fold,
                            'Model': model_name,
                            'Train_Time_Sec': round(train_time, 4),
                            'Avg_Single_Pred_Time_ms': metrics['avg_time'],
                            'Std_Single_Pred_Time_ms': metrics['std_time'],
                            'Imbalance_Ratio': round(imb_ratio, 2),
                            'Accuracy': metrics['accuracy'],
                            'Precision': metrics['precision'],
                            'Recall': metrics['recall'],
                            'F1_Score': metrics['f1_score'],
                            'Sensitivity': metrics['sensitivity'],
                            'Specificity': metrics['specificity'],
                            'FNR': metrics['fnr'],
                            'FPR': metrics['fpr'],
                            'TN': metrics['TN'],
                            'FP': metrics['FP'],
                            'FN': metrics['FN'],
                            'TP': metrics['TP'],
                            'Fit_Mode': fit_mode
                        }
                        
                        # Save record
                        self._upsert_record(record)
                        print(f"    ✓ {model_name}: Acc={metrics['accuracy']:.4f}, "
                              f"F1={metrics['f1_score']:.4f} ({train_time:.1f}s)")
                        
                        # Update progress
                        fold_pbar.update(1)
                        
                        # Save checkpoint after each model
                        try:
                            self.checkpoint.save_checkpoint(
                                self.full_report_data, self.completed_folds
                            )
                        except Exception as e:
                            print(f"[CHECKPOINT] Save error: {e}")
            
            # Mark fold as completed
            try:
                self.completed_folds.add(fold)
                self.checkpoint.save_checkpoint(
                    self.full_report_data, self.completed_folds
                )
                print(f"[CHECKPOINT] Fold {fold} completed and saved.")
            except Exception as e:
                print(f"[CHECKPOINT] Error finalizing fold {fold}: {e}")
        
        return self.full_report_data
        
    def run_holdout_evaluation(self, train_indices, holdout_indices, labels):
        """Run final evaluation on holdout set and ablation study"""
        print("\n" + "="*80)
        print("=== FINAL HOLDOUT EVALUATION ===")
        print("="*80)
        
        # Build features for final train and holdout sets
        self.feature_builder.store.clear()
        self.feature_builder.build_counters_from_train(
            train_indices, labels, threshold=self.config.feature_threshold
        )
        
        print("Building X_train_final...")
        X_train_final = self.feature_builder.build_features(train_indices)
        print("Building X_test_holdout...")
        X_test_holdout = self.feature_builder.build_features(holdout_indices)
        
        y_train_final = labels[train_indices]
        y_test_holdout = labels[holdout_indices]
        
        print(f"Final train shape: {X_train_final.shape}")
        print(f"Holdout test shape: {X_test_holdout.shape}")
        
        # Final Model Training and Evaluation (Baseline)
        final_results = []
        holdout_predictions = {}
        
        for model_name, model in self.models.items():
            print(f"\nTraining final model: {model_name}")
            clf = clone(model)
            clf, fit_mode = self.train_model(clf, X_train_final, y_train_final)
            
            # Predict
            y_pred = self.evaluator.predict_with_progress(clf, X_test_holdout, model_name=model_name)
            holdout_predictions[model_name] = y_pred
            
            # Calculate metrics
            metrics = self.evaluator.calculate_metrics(y_test_holdout, y_pred)
            print(f"{model_name} HOLDOUT -> Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1_score']:.4f}")
            
            final_results.append({
                "Model": model_name,
                "Accuracy": round(metrics['accuracy'], 6),
                "Precision": round(metrics['precision'], 6),
                "Recall": round(metrics['recall'], 6),
                "F1": round(metrics['f1_score'], 6),
                "Sensitivity": round(metrics['sensitivity'], 6),
                "Specificity": round(metrics['specificity'], 6),
                "FNR": round(metrics['fnr'], 6),
                "FPR": round(metrics['fpr'], 6),
                "TN": metrics['TN'],
                "FP": metrics['FP'],
                "FN": metrics['FN'],
                "TP": metrics['TP'],
                "Fit_Mode": fit_mode
            })
            
        # Holdout Ablation Study
        print("\n" + "="*80)
        print("=== HOLDOUT ABLATION STUDY ===")
        print("="*80)
        
        scenarios = {
            'BASELINE': [],
            'REMOVE_BoW': [14],
            'REMOVE_SegBoW': [15],
            'REMOVE_NGRAMS_3_4': [16,17],
            'REMOVE_TLD': [3],
            'REMOVE_RATIOS': [2,18,19]
        }
        
        def drop_columns_safe(X, cols_to_drop):
            if X is None or X.size == 0 or not cols_to_drop:
                return X
            valid = [c for c in cols_to_drop if 0 <= c < X.shape[1]]
            return np.delete(X, valid, axis=1) if valid else X
            
        holdout_ablation_data = []
        
        for scen_name, remove_cols in scenarios.items():
            print(f"\n[ABL] Scenario: {scen_name}")
            X_train_scen = drop_columns_safe(X_train_final.copy(), remove_cols)
            X_test_scen  = drop_columns_safe(X_test_holdout.copy(), remove_cols)
            
            for model_name, model in self.models.items():
                clf = clone(model)
                t0 = time.time()
                clf, fit_mode = self.train_model(clf, X_train_scen, y_train_final)
                train_time = time.time() - t0
                
                y_pred = self.evaluator.predict_with_progress(clf, X_test_scen, model_name=model_name)
                m = self.evaluator.calculate_metrics(y_test_holdout, y_pred)
                
                holdout_ablation_data.append({
                    'Scenario': scen_name,
                    'Model': model_name,
                    'Accuracy': round(m['accuracy'], 6),
                    'Precision': round(m['precision'], 6),
                    'Recall': round(m['recall'], 6),
                    'F1_Score': round(m['f1_score'], 6),
                    'Sensitivity': round(m['sensitivity'], 6),
                    'Specificity': round(m['specificity'], 6),
                    'FNR': round(m['fnr'], 6),
                    'FPR': round(m['fpr'], 6),
                    'TN': m['TN'], 'FP': m['FP'], 'FN': m['FN'], 'TP': m['TP'],
                    'Train_Time_Sec': round(train_time, 4),
                    'Fit_Mode': fit_mode
                })
                print(f"   ✓ {model_name}: Acc={m['accuracy']:.4f}, F1={m['f1_score']:.4f}")
                
        return final_results, holdout_predictions, holdout_ablation_data
