"""
URL Phishing Detection - Ensemble Deep Learning Framework

A modular implementation of ensemble deep learning models for URL phishing detection,
featuring comprehensive feature engineering and statistical validation.
"""

__version__ = "1.0.0"
__author__ = "Deniz Kaya"

from .preprocessing import (
    prepare_data_from_raw,
    tokenize_and_pad_urls,
    scale_numerical_features
)

from .model import (
    create_model_architecture,
    compile_model,
    ModelArchitecture
)

from .evaluation import (
    calculate_metrics,
    evaluate_model,
    print_metrics,
    plot_confusion_matrix,
    compare_models
)

from .statistical_tests import (
    StatisticalSignificanceAnalyzer,
    run_statistical_tests
)

from .feature_extraction import (
    VectorizedFeatureExtractor,
    IsolatedFeatureManager,
    extract_features_for_fold_optimized
)

from .ensemble_classifier import (
    OptimizedEnsembleURLClassifierCV
)

__all__ = [
    # Preprocessing
    'prepare_data_from_raw',
    'tokenize_and_pad_urls',
    'scale_numerical_features',
    
    # Model
    'create_model_architecture',
    'compile_model',
    'ModelArchitecture',
    
    # Evaluation
    'calculate_metrics',
    'evaluate_model',
    'print_metrics',
    'plot_confusion_matrix',
    'compare_models',
    
    # Statistical Tests
    'StatisticalSignificanceAnalyzer',
    'run_statistical_tests',
    
    # Feature Extraction
    'VectorizedFeatureExtractor',
    'IsolatedFeatureManager',
    'extract_features_for_fold_optimized',
    
    # Ensemble Classifier
    'OptimizedEnsembleURLClassifierCV',
]
