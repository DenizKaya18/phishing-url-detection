import sys
import os
import time
import numpy as np
from src.config import *
from src.utils import Logger, format_time
from src.classifier import OptimizedEnsembleURLClassifierCV
from src.models import CheckpointManager

# Redirect all stdout to both terminal and a result file in results/ directory
sys.stdout = Logger()

def main():
    """
    Main execution pipeline for URL Phishing Detection.
    Coordinates data loading, cross-validation, final training, and detailed reporting.
    """
    start_total_time = time.time()
    
    print("="*80)
    print("🚀 URL PHISHING DETECTION SYSTEM - MODULAR VERSION")
    print("="*80)
    print(f"Working Directory: {BASE_DIR}")
    print(f"Data Path: {DATA_PATH}")
    print(f"Results will be saved to: {RESULTS_DIR}")
    print("-" * 80)

    # 1. Initialize Classifier and Checkpoint Manager
    ensemble_clf = OptimizedEnsembleURLClassifierCV(
        n_models=N_MODELS,
        n_folds=N_FOLDS,
        random_seeds=RANDOM_SEEDS
    )
    
    checkpoint_mgr = CheckpointManager(checkpoint_dir=CHECKPOINT_DIR)

    # 2. Data Preparation
    if not os.path.exists(DATA_PATH):
        print(f"✗ Error: Dataset not found at {DATA_PATH}")
        return

    X_url_train, y_train, X_url_test, y_test = ensemble_clf.prepare_data_from_raw(
        DATA_PATH, test_size=0.2
    )

    # 3. PHASE 1: Cross-Validation with Checkpoints
    print("\n" + "="*80)
    print("📋 PHASE 1: STARTING CROSS-VALIDATION")
    print("="*80)
    
    ensemble_clf.cross_validate_ensemble(
        X_url_train, 
        y_train, 
        checkpoint_mgr=checkpoint_mgr,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )

    # 4. PHASE 2: Final Ensemble Training
    print("\n" + "="*80)
    print("📋 PHASE 2: TRAINING FINAL ENSEMBLE")
    print("="*80)
    
    ensemble_clf.train_final_ensemble(
        X_url_train, 
        y_train, 
        X_url_test, 
        y_test,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )
    
    # 5. PHASE 3: Evaluation & Detailed Reporting
    # This generates the COMPREHENSIVE PERFORMANCE REPORT and TIME STATISTICS
    ensemble_results, best_method = ensemble_clf.evaluate_final_ensemble(X_url_test, y_test)
    ensemble_clf.print_comprehensive_summary(ensemble_results, best_method)
    
    # Orijinal Log: COMPREHENSIVE CONFUSION MATRIX AND METRICS
    print("\n" + "="*80)
    print("?? COMPREHENSIVE CONFUSION MATRIX AND PERFORMANCE METRICS REPORT")
    print("="*80)
    
    # CV Average Results
    print("\n" + "="*80)
    print("? CROSS-VALIDATION RESULTS (Average across 10 Folds)")
    print("="*80)
    if ensemble_clf.cv_confusion_matrices:
        avg_cm = np.mean(ensemble_clf.cv_confusion_matrices, axis=0)
        ensemble_clf.print_confusion_matrix_detailed(avg_cm, "Average Confusion Matrix (CV)")
    
    # Detailed Comparison Table (CV vs Final Test)
    ensemble_clf.print_cv_comparison(ensemble_results, best_method)

    # Fold-by-Fold Metrics breakdown
    print("\n" + "="*80)
    print("📋 DETAILED FOLD-BY-FOLD METRICS")
    print("="*80)
    if hasattr(ensemble_clf, 'cv_metrics') and ensemble_clf.cv_metrics:
        for i, m in enumerate(ensemble_clf.cv_metrics):
            ensemble_clf.print_metrics_detailed(m, f"Fold {i+1} Metrics")

    # 6. PHASE 4: Latency Summary
    # Measures how fast the system processes a single URL
    ensemble_clf.measure_single_url_latency(X_url_test, n_samples=100)

    # 7. PHASE 5: Statistical Significance Testing
    print("\n" + "="*80)
    print("📊 PHASE 5: STATISTICAL SIGNIFICANCE TESTING")
    print("="*80)
    try:
        from src.statistical_tests import run_statistical_analysis
        analyzer = run_statistical_analysis(ensemble_clf)
        # Add ANOVA comparison from logs
        analyzer.run_anova_all_models()
    except Exception as e:
        print(f"[!] Statistical analysis skipped or failed: {e}")
    
    # 8. PHASE 6: Save Final Models and Configuration
    print("\n" + "="*80)
    print("📋 PHASE 6: SAVING MODELS AND RESULTS")
    print("="*80)
    
    model_save_path = os.path.join(MODELS_DIR, "ensemble_final")
    ensemble_clf.save_model_ensemble(save_path=model_save_path)

    # Final Summary Execution
    total_duration = time.time() - start_total_time
    print("\n" + "="*80)
    print("✅ PROCESS COMPLETED SUCCESSFULLY")
    print(f"Total System Execution Time: {format_time(total_duration)}")
    print(f"Final models saved in: {MODELS_DIR}")
    print(f"Detailed logs saved in: {RESULTS_DIR}")
    print("="*80)

def extend_classifier_functionality():
    """
    Helper function to inject utility functions into the class dynamically.
    Ensures model saving logic is present without cluttering classifier.py.
    """
    import pickle
    
    def save_model_ensemble(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        # Save pre-processing tools (Tokenizer and Scaler)
        with open(os.path.join(save_path, "tokenizer.pkl"), "wb") as f:
            pickle.dump(self.tokenizer, f)
        with open(os.path.join(save_path, "scaler.pkl"), "wb") as f:
            pickle.dump(self.scaler, f)

        # Save model architectures and info
        config = {
            'vocab_size': self.vocab_size,
            'max_len': self.max_len,
            'n_models': self.n_models,
            'random_seeds': self.random_seeds,
            'training_finished_at': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(os.path.join(save_path, "config.pkl"), "wb") as f:
            pickle.dump(config, f)

        # Save each model in the ensemble as a .keras file
        for i, model in enumerate(self.models):
            model.save(os.path.join(save_path, f"model_{i+1}.keras"))
        
        print(f"✓ All models and configuration successfully exported to {save_path}")

    # Inject the function into the class definition at runtime
    OptimizedEnsembleURLClassifierCV.save_model_ensemble = save_model_ensemble

if __name__ == "__main__":
    # Apply injections and start the main process
    extend_classifier_functionality()
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ SYSTEM INTERRUPTED BY USER")
        print("✓ Progress state maintained in cv_checkpoints/ directory.")
    except Exception as e:
        print(f"\n✗ CRITICAL SYSTEM ERROR: {e}")
        import traceback
        traceback.print_exc()