# -*- coding: utf-8 -*-
"""
Model Evaluation Module
"""

import time
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
from tqdm import tqdm


class ModelEvaluator:
    """Handles model evaluation and metrics calculation"""
    
    def __init__(self, config):
        self.config = config
    
    def predict_with_progress(self, clf, X, model_name="Model"):
        """Predict in batches with progress bar"""
        n = len(X)
        batch_size = self.config.prediction_batch_size
        preds = []
        
        for start in tqdm(range(0, n, batch_size), desc=f"Predicting {model_name}", leave=False):
            end = min(start + batch_size, n)
            batch = X[start:end]
            batch_preds = clf.predict(batch)
            preds.extend(batch_preds)
        
        return np.array(preds)
    
    def measure_prediction_time(self, clf, X):
        """Measure single prediction time statistics"""
        times = []
        n_sample = min(self.config.timing_sample_size, len(X))
        
        if n_sample > 0:
            subset = X[:n_sample]
            for row in subset:
                ts = time.perf_counter()
                clf.predict(row.reshape(1, -1))
                times.append((time.perf_counter() - ts) * 1000.0)  # Convert to ms
            
            avg_time = float(np.mean(times))
            std_time = float(np.std(times))
        else:
            avg_time = 0.0
            std_time = 0.0
        
        return round(avg_time, 6), round(std_time, 6)
    
    def calculate_metrics(self, y_true, y_pred):
        """Calculate all performance metrics"""
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        cm = confusion_matrix(y_true, y_pred)
        
        # Safe confusion matrix parsing
        if cm.size == 4 and cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        elif cm.size == 1 and cm.shape == (1, 1):
            if np.all(y_true == 1):
                tp = cm[0, 0]
                tn = fp = fn = 0
            else:
                tn = cm[0, 0]
                tp = fp = fn = 0
        else:
            tn = fp = fn = tp = 0
        
        # Calculate additional metrics
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        fnr = fn / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        return {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'sensitivity': sens,
            'specificity': spec,
            'fnr': fnr,
            'fpr': fpr,
            'TN': int(tn),
            'FP': int(fp),
            'FN': int(fn),
            'TP': int(tp)
        }
    
    def evaluate_model(self, clf, X_test, y_test, model_name):
        """Evaluate model and return all metrics"""
        # Measure prediction time
        avg_time, std_time = self.measure_prediction_time(clf, X_test)
        
        # Get predictions
        y_pred = self.predict_with_progress(clf, X_test, model_name)
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_test, y_pred)
        metrics['avg_time'] = avg_time
        metrics['std_time'] = std_time
        
        return metrics
