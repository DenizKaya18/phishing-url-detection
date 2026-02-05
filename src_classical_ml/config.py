# -*- coding: utf-8 -*-
"""
Configuration Management for Classical ML Pipeline
"""

import os
import multiprocessing


class Config:
    """Configuration class for classical ML pipeline"""
    
    def __init__(self, output_folder=None, data_file=None):
        # Get the directory of this config file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Paths
        if output_folder is None:
            # Create results folder in the same directory as the script
            self.output_folder = os.path.join(current_dir, "results")
        else:
            self.output_folder = output_folder
            
        if data_file is None:
            self.data_file = os.path.join(current_dir, "..", "data", "dataset.txt")
        else:
            self.data_file = data_file
        
        # System settings
        self.n_jobs = max(1, multiprocessing.cpu_count() - 1)
        self.joblib_backend = 'loky'
        self.joblib_verbose = 10
        
        # Environment variables for thread control
        os.environ["OMP_NUM_THREADS"] = str(self.n_jobs)
        os.environ["OPENBLAS_NUM_THREADS"] = str(self.n_jobs)
        os.environ["MKL_NUM_THREADS"] = str(self.n_jobs)
        
        # Cross-validation settings
        self.n_splits = 10
        self.random_state = 42
        
        # Feature extraction settings
        self.feature_threshold = 20
        
        # Checkpoint settings
        self.partial_csv_name = "Detailed_Performance_Report_Per_Fold_partial.csv"
        self.completed_meta_name = "completed_folds.json"
        
        # Cache settings
        self.cache_path = os.path.join(self.output_folder, "preprocessed_cache_v1.pkl")
        
        # Batch settings
        self.preprocessing_batch_size = 5000
        self.prediction_batch_size = 5000
        self.timing_sample_size = 100
        
    def ensure_directories(self):
        """Create necessary directories"""
        os.makedirs(self.output_folder, exist_ok=True)
        
    def __str__(self):
        return f"Config(n_jobs={self.n_jobs}, n_splits={self.n_splits})"