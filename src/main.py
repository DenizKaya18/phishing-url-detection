"""
Main script for training and evaluating ensemble URL phishing detection models.

This script demonstrates the end-to-end pipeline:
1. Data preprocessing
2. Model training with cross-validation
3. Final ensemble evaluation
4. Statistical significance testing
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

from preprocessing import prepare_data_from_raw
from ensemble_classifier import OptimizedEnsembleURLClassifierCV
from evaluation import evaluate_model, calculate_cv_statistics
from statistical_tests import run_statistical_tests


def main(data_file="data/dataset.txt", 
         test_size=0.2,
         n_folds=10,
         epochs=15,
         batch_size=512):
    """
    Main training and evaluation pipeline.
    
    Args:
        data_file: Path to raw data file
        test_size: Fraction of data for testing
        n_folds: Number of cross-validation folds
        epochs: Training epochs per model
        batch_size: Training batch size
    """
    
    print("\n" + "="*80)
    print("URL PHISHING DETECTION - ENSEMBLE DEEP LEARNING")
    print("="*80)
    
    # ==================== 1. DATA PREPROCESSING ====================
    print("\n📂 STEP 1: Data Preprocessing")
    print("-" * 80)
    
    X_url_train, X_url_test, y_train, y_test, tokenizer, max_len, vocab_size = \
        prepare_data_from_raw(data_file, test_size=test_size)
    
    print(f"✓ Data prepared successfully")
    print(f"  Training samples: {len(X_url_train)}")
    print(f"  Test samples: {len(X_url_test)}")
    
    # ==================== 2. INITIALIZE CLASSIFIER ====================
    print("\n🔬 STEP 2: Initialize Ensemble Classifier")
    print("-" * 80)
    
    classifier = OptimizedEnsembleURLClassifierCV(
        n_models=4,
        n_folds=n_folds,
        random_seeds=[42, 123, 456, 789]
    )
    
    # Set tokenizer and vocab info
    classifier.tokenizer = tokenizer
    classifier.max_len = max_len
    classifier.vocab_size = vocab_size
    
    # ==================== 3. CROSS-VALIDATION ====================
    print("\n📊 STEP 3: Cross-Validation Training")
    print("-" * 80)
    
    cv_individual, cv_ensemble = classifier.cross_validate_ensemble(
        X_url_train, y_train,
        epochs=epochs,
        batch_size=batch_size
    )
    
    # Print CV results
    print("\n📈 Cross-Validation Results:")
    print("-" * 80)
    for model_type, scores in cv_individual.items():
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        print(f"  {model_type:12}: {mean_score:.4f} (± {std_score:.4f})")
    
    ensemble_mean = np.mean(cv_ensemble)
    ensemble_std = np.std(cv_ensemble)
    print(f"  {'Ensemble':12}: {ensemble_mean:.4f} (± {ensemble_std:.4f})")
    
    # ==================== 4. FINAL TRAINING ====================
    print("\n🎯 STEP 4: Final Ensemble Training")
    print("-" * 80)
    
    classifier.train_final_ensemble(
        X_url_train, y_train,
        X_url_test, y_test,
        epochs=epochs,
        batch_size=batch_size
    )
    
    # ==================== 5. STATISTICAL TESTS ====================
    print("\n📈 STEP 5: Statistical Significance Testing")
    print("-" * 80)
    
    try:
        statistical_results = run_statistical_tests(classifier)
        print("✓ Statistical tests completed")
    except Exception as e:
        print(f"⚠️ Statistical tests error: {e}")
        statistical_results = None
    
    # ==================== 6. SUMMARY ====================
    print("\n" + "="*80)
    print("📋 FINAL SUMMARY")
    print("="*80)
    
    print(f"\n✓ Cross-Validation Performance:")
    print(f"  Ensemble Mean: {ensemble_mean:.4f}")
    print(f"  Ensemble Std:  {ensemble_std:.4f}")
    
    print(f"\n✓ Model Performance:")
    for i, info in enumerate(classifier.model_info):
        print(f"  Model {i+1} ({info['type']:12}): {info['test_accuracy']:.4f}")
    
    print(f"\n✓ Training Time: {classifier.format_time(classifier.total_training_time)}")
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETED SUCCESSFULLY")
    print("="*80)
    
    return classifier, statistical_results


def quick_example():
    """
    Quick example showing basic usage.
    """
    print("\n" + "="*80)
    print("QUICK USAGE EXAMPLE")
    print("="*80)
    
    print("\n1️⃣ Import modules:")
    print("   from preprocessing import prepare_data_from_raw")
    print("   from ensemble_classifier import OptimizedEnsembleURLClassifierCV")
    print("   from evaluation import evaluate_model")
    
    print("\n2️⃣ Load data:")
    print("   X_train, X_test, y_train, y_test, tokenizer, max_len, vocab = \\")
    print("       prepare_data_from_raw('data/dataset.txt')")
    
    print("\n3️⃣ Initialize classifier:")
    print("   classifier = OptimizedEnsembleURLClassifierCV(n_models=4, n_folds=10)")
    print("   classifier.tokenizer = tokenizer")
    print("   classifier.max_len = max_len")
    print("   classifier.vocab_size = vocab")
    
    print("\n4️⃣ Train with cross-validation:")
    print("   cv_scores = classifier.cross_validate_ensemble(X_train, y_train)")
    
    print("\n5️⃣ Train final ensemble:")
    print("   classifier.train_final_ensemble(X_train, y_train, X_test, y_test)")
    
    print("\n6️⃣ Make predictions:")
    print("   y_pred, y_proba = classifier.predict_ensemble(X_test_urls, X_test_features)")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--example":
        # Show example
        quick_example()
    else:
        # Run full pipeline
        try:
            classifier, stats = main()
            print("\n✓ All analyses completed successfully!")
        except KeyboardInterrupt:
            print("\n\n⚠️ Training interrupted by user")
        except Exception as e:
            print(f"\n✗ Error: {e}")
            import traceback
            traceback.print_exc()

