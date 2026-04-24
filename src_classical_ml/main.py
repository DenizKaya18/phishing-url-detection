# -*- coding: utf-8 -*-
"""
Main Script for Classical ML Pipeline
"""

import sys
import os
import warnings

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from sklearn.model_selection import train_test_split

from config import Config
from data_loader import DataLoader
from preprocessor import URLPreprocessor
from feature_builder import FeatureBuilder
from models import ModelFactory
from trainer import ModelTrainer
from report_generator import ReportGenerator


def setup_environment():
    """Setup environment"""
    print("="*80)
    print("=== ENVIRONMENT SETUP ===")
    print("="*80)
    
    # Set recursion limit and warnings
    sys.setrecursionlimit(21000)
    warnings.filterwarnings('ignore')
    
    print("[INFO] Environment setup complete\n")


def main():
    """Main pipeline execution"""
    # 1. Setup environment
    setup_environment()
    
    # 2. Initialize configuration
    config = Config()
    config.ensure_directories()
    print(f"[INFO] Configuration: {config}")
    print(f"[INFO] Output folder: {config.output_folder}\n")
    
    # 3. Load data
    data_loader = DataLoader(config.data_file)
    urls, labels = data_loader.load_data()
    
    # 4. Preprocess URLs
    preprocessor = URLPreprocessor(config)
    preprocessed_data = preprocessor.preprocess_all(urls)
    valid_indices, valid_labels = preprocessor.filter_valid_samples(labels)
    
    # 5. Initialize feature builder
    feature_builder = FeatureBuilder(config, preprocessed_data)
    
    # 6. Get models
    models = ModelFactory.get_models()
    print(f"\n[INFO] Models to train: {list(models.keys())}")
    
    # 7. Split Data (Holdout)
    print("\n" + "="*80)
    print("=== HOLDOUT SPLIT ===")
    print("="*80)
    train_indices, holdout_indices = train_test_split(
        valid_indices,
        test_size=config.test_size,
        stratify=valid_labels,
        random_state=config.random_state
    )
    
    print(f"Train size : {len(train_indices)}")
    print(f"Test size  : {len(holdout_indices)}")
    import numpy as np
    # To get distribution, we map train_indices back to valid_labels index
    # valid_indices is an array-like of the original indices.
    # We need the labels corresponding to train_indices and holdout_indices
    # Actually train_test_split on indices directly needs us to map them to labels.
    # We should split on valid_indices, and we can stratify by valid_labels.
    # We will need the corresponding labels for bincount.
    train_labels = [labels[i] for i in train_indices]
    holdout_labels = [labels[i] for i in holdout_indices]
    print(f"Train dist : {np.bincount(train_labels)}")
    print(f"Test dist  : {np.bincount(holdout_labels)}")
    
    # 8. Train models with cross-validation
    trainer = ModelTrainer(config, feature_builder, models)
    cv_results = trainer.run_cross_validation(
        train_indices,
        labels,
        labels
    )
    
    # 9. Final Holdout Evaluation
    holdout_results, holdout_predictions, ablation_data = trainer.run_holdout_evaluation(
        train_indices,
        holdout_indices,
        labels
    )
    
    # 10. Generate reports
    report_gen = ReportGenerator(config)
    report_gen.generate_reports(cv_results)
    report_gen.generate_holdout_reports(holdout_results, holdout_predictions, ablation_data, holdout_indices, labels)
    
    print("\n" + "="*80)
    print("=== PIPELINE COMPLETED SUCCESSFULLY ===")
    print("="*80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Pipeline interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
