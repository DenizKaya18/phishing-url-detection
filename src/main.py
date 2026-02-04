"""
Main entry point for URL Phishing Detection Pipeline.
Replicates Google Colab environment settings and execution flow.
"""

import sys
import tensorflow as tf
from tensorflow.keras import mixed_precision
import numpy as np
import time

# 1. System Settings
sys.setrecursionlimit(21000)

# 2. GPU & Mixed Precision Setup
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✓ GPU Memory Growth Enabled: {len(gpus)} GPU(s)")
        
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
        print("✓ Mixed Precision (mixed_float16) Enabled")
    except RuntimeError as e:
        print(e)

from src.preprocessing import prepare_data_from_raw
from src.ensemble_classifier import OptimizedEnsembleURLClassifierCV, CheckpointManager
from src.statistical_tests import run_statistical_tests

def main():
    print("="*80)
    print("URL PHISHING DETECTION - ENSEMBLE DEEP LEARNING (OPTIMIZED)")
    print("="*80)

    # Checkpoint Manager
    checkpoint_mgr = CheckpointManager(storage_type="local")

    # Data Loading
    RAW_DATA_FILE = "data/dataset.txt" # Adjust path as needed
    
    classifier = OptimizedEnsembleURLClassifierCV(
        n_models=4,
        n_folds=3,
        random_seeds=[42, 123, 456, 789]
    )

    # 1. Prepare Data
    (X_url_train, y_train, X_url_test, y_test, rows_all, tokenizer, max_len, vocab_size) = \
        prepare_data_from_raw(RAW_DATA_FILE, test_size=0.2)

    rows_train = [(X_url_train[i], y_train[i]) for i in range(len(X_url_train))]
    rows_test = [(X_url_test[i], y_test[i]) for i in range(len(X_url_test))]

    # 2. Cross Validation
    print("\n>>> Starting Cross-Validation...")
    classifier.cross_validate_ensemble(
        X_url_train, y_train, rows_train,
        checkpoint_mgr=checkpoint_mgr,
        epochs=3,
        batch_size=512
    )
    
    # 3. Efficiency Report
    classifier.print_training_efficiency_summary()
    classifier.export_efficiency_report("training_efficiency_report.csv")

    # 4. Final Training
    print("\n>>> Starting Final Ensemble Training...")
    classifier.train_final_ensemble(
        X_url_train, y_train, rows_train,
        X_url_test, y_test, rows_test,
        epochs=3,
        batch_size=512
    )

    # 5. Evaluation
    print("\n>>> Final Evaluation...")
    classifier.evaluate_final_ensemble()
    classifier.print_comprehensive_summary()
    classifier.print_confusion_matrix_and_metrics()
    
    # 6. Save Model
    classifier.save_model_ensemble("models/")

    # 7. Latency & Ablation
    classifier.measure_single_url_latency(X_url_test, n_samples=100)
    classifier.run_ablation_study_final(X_url_train, y_train, X_url_test, y_test)

    # 8. Stats
    try:
        run_statistical_tests(classifier)
    except Exception as e:
        print(f"Stats error: {e}")

if __name__ == "__main__":
    main()