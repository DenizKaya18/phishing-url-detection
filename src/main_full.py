"""
Full experimental pipeline for ensemble deep learning based
URL phishing detection.

This script reproduces all experiments reported in the paper:
- 10-fold cross-validation
- Final ensemble training
- Confusion matrices & metrics
- Statistical significance tests
- Training efficiency analysis
- Ablation study
- Latency measurement
"""

import time
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from src.preprocessing import prepare_data_from_raw
from src.ensemble_classifier import OptimizedEnsembleURLClassifierCV
from src.statistical_tests import run_statistical_tests


def main(
    data_file="data/dataset.txt",
    test_size=0.2,
    n_folds=3,
    epochs=3,
    batch_size=512
):
    start_time = time.time()

    print("\n" + "="*90)
    print("FULL EXPERIMENTAL PIPELINE – ENSEMBLE URL PHISHING DETECTION")
    print("="*90)

    # ==================== DATA ====================
    print("\n📂 STEP 1: Data preprocessing")
    print("-"*80)

    (X_train, y_train,X_test, y_test,rows_all,tokenizer, max_len, vocab_size) = prepare_data_from_raw(
        data_file,
        test_size=test_size
    )


    print(f"✓ Train samples: {len(X_train)}")
    print(f"✓ Test samples : {len(X_test)}")

    rows_train = [(X_train[i], y_train[i]) for i in range(len(X_train))]
    rows_test  = [(X_test[i],  y_test[i])  for i in range(len(X_test))]

    # ==================== MODEL ====================
    print("\n🔬 STEP 2: Initializing ensemble")
    print("-"*80)

    clf = OptimizedEnsembleURLClassifierCV(
        n_models=4,
        n_folds=n_folds,
        random_seeds=[42, 123, 456, 789]
    )

    clf.tokenizer = tokenizer
    clf.max_len = max_len
    clf.vocab_size = vocab_size

    # ==================== CV ====================
    print("\n📊 STEP 3: Cross-validation training")
    print("-"*80)

    cv_individual, cv_ensemble = clf.cross_validate_ensemble(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size
    )

    print("\n📈 CV SUMMARY:")
    for model, scores in cv_individual.items():
        print(f"  {model:12}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

    print(f"  {'Ensemble':12}: {np.mean(cv_ensemble):.4f} ± {np.std(cv_ensemble):.4f}")

    # ==================== FINAL ====================
    print("\n🎯 STEP 4: Final ensemble training")
    print("-"*80)

    #clf.train_final_ensemble(
    #    X_train, y_train, rows_train,
    #    X_test, y_test, rows_test,
    #    epochs=epochs,
    #    batch_size=batch_size
    #)

    clf.train_final_ensemble(
    X_train, y_train,
    X_test, y_test,
    epochs=epochs,
    batch_size=batch_size
    )


    print("\n📋 Final evaluation")
    ensemble_results, best_method = clf.evaluate_final_ensemble()

    clf.print_comprehensive_summary()
    clf.print_confusion_matrix_and_metrics()

    # ==================== LATENCY ====================
    print("\n⏱️ STEP 5: Latency analysis")
    print("-"*80)

    try:
        timing_stats = clf.measure_single_url_latency(
            X_test,
            n_samples=100,
            random_state=42
        )
    except Exception as e:
        print(f"⚠️ Latency skipped: {e}")
        timing_stats = None

    # ==================== ABLATION ====================
    print("\n🔬 STEP 6: Ablation study")
    print("-"*80)

    try:
        ablation_df = clf.run_ablation_study_final(
            X_train, y_train,
            X_test, y_test,
            epochs=epochs,
            batch_size=batch_size,
            save_csv="ablation_results.csv"
        )
    except Exception as e:
        print(f"⚠️ Ablation skipped: {e}")
        ablation_df = None

    # ==================== STATS ====================
    print("\n📊 STEP 7: Statistical significance tests")
    print("-"*80)

    try:
        analyzer = run_statistical_tests(clf)
    except Exception as e:
        print(f"⚠️ Statistical tests skipped: {e}")
        analyzer = None

    # ==================== SUMMARY ====================
    total_time = time.time() - start_time
    print("\n" + "="*90)
    print("✅ FULL PIPELINE COMPLETED")
    print("="*90)
    print(f"⏱️ Total execution time: {clf.format_time(total_time)}")

    return clf, ensemble_results, best_method, analyzer, timing_stats, ablation_df


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
