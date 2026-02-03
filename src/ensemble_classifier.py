"""
Ensemble classifier for URL phishing detection.
Combines multiple deep learning architectures with cross-validation.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score
import time
import gc

from .model import create_model_architecture, compile_model
from .evaluation import calculate_metrics
from .feature_extraction import IsolatedFeatureManager, extract_features_for_fold_optimized


class OptimizedEnsembleURLClassifierCV:
    """
    Optimized ensemble URL classifier with cross-validation support.
    """

    def __init__(self, n_models=4, random_seeds=None, n_folds=10):
        """
        Initialize ensemble classifier.
        
        Args:
            n_models: Number of models in ensemble
            random_seeds: List of random seeds for each model
            n_folds: Number of cross-validation folds
        """
        self.n_models = n_models
        self.random_seeds = random_seeds or [42, 123, 456, 789][:n_models]
        self.n_folds = n_folds
        
        # Model storage
        self.models = []
        self.histories = []
        self.scaler = None
        self.tokenizer = None
        self.max_len = None
        self.vocab_size = None
        
        # Training info
        self.training_times = []
        self.model_info = []
        self.total_training_time = 0
        
        # Cross-validation results
        self.cv_scores = {}
        self.cv_detailed_results = []
        self.cv_confusion_matrices = []
        self.cv_metrics = []
        
        # Final test results
        self.final_confusion_matrix = None
        self.final_metrics = None
        self.avg_confusion_matrix = None
        self.avg_metrics = None
        
        # Feature manager
        self.feature_manager = IsolatedFeatureManager()
        
        print(f"✓ Ensemble classifier initialized: {n_models} models, {n_folds} folds")

    def cleanup_gpu_memory(self, fold_idx=None):
        """Clean up GPU memory."""
        try:
            tf.keras.backend.clear_session()
            gc.collect()
            
            if fold_idx is not None:
                print(f"✓ Fold {fold_idx+1} GPU memory cleaned")
        except:
            pass

    def format_time(self, seconds):
        """Format time in human-readable format."""
        if seconds < 60:
            return f"{seconds:.2f} seconds"
        elif seconds < 3600:
            return f"{seconds/60:.2f} minutes"
        else:
            return f"{seconds/3600:.2f} hours"

    def create_optimized_callbacks(self, model_name):
        """
        Create training callbacks.
        
        Args:
            model_name: Name for callback identification
        
        Returns:
            List of callbacks
        """
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,
                patience=2,
                min_lr=1e-7,
                verbose=1
            )
        ]
        return callbacks

    def train_model_fold(self, X_url_train, X_num_train, y_train,
                        X_url_val, X_num_val, y_val,
                        model_type, seed, fold_idx, class_weight_dict,
                        epochs=15, batch_size=128):
        """
        Train a single model for one fold.
        
        Args:
            X_url_train: Training URLs (processed)
            X_num_train: Training numerical features
            y_train: Training labels
            X_url_val: Validation URLs (processed)
            X_num_val: Validation numerical features
            y_val: Validation labels
            model_type: Type of model architecture
            seed: Random seed
            fold_idx: Fold index for logging
            class_weight_dict: Class weights
            epochs: Training epochs
            batch_size: Batch size
        
        Returns:
            Trained model, history, metrics, and scaler
        """
        model_train_start = time.time()
        
        # Scale numerical features
        scaler = StandardScaler()
        X_num_train_scaled = scaler.fit_transform(X_num_train)
        X_num_val_scaled = scaler.transform(X_num_val)
        
        # Tokenize and pad URLs
        seq_train = self.tokenizer.texts_to_sequences(X_url_train)
        seq_val = self.tokenizer.texts_to_sequences(X_url_val)
        
        X_url_train_pad = pad_sequences(seq_train, maxlen=self.max_len,
                                       padding="post", truncating="post")
        X_url_val_pad = pad_sequences(seq_val, maxlen=self.max_len,
                                     padding="post", truncating="post")
        
        # Create and compile model
        model = create_model_architecture(
            self.vocab_size, self.max_len, X_num_train_scaled.shape[1],
            model_type=model_type, seed=seed
        )
        model = compile_model(model)
        
        # Callbacks
        callbacks = self.create_optimized_callbacks(f"fold_{fold_idx}_{model_type}")
        
        # Train
        history = model.fit(
            x={"url_input": X_url_train_pad, "num_input": X_num_train_scaled},
            y=y_train,
            validation_data=(
                {"url_input": X_url_val_pad, "num_input": X_num_val_scaled}, y_val
            ),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight_dict,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate
        val_pred_proba = model.predict(
            {"url_input": X_url_val_pad, "num_input": X_num_val_scaled},
            verbose=0
        ).flatten()
        val_pred = (val_pred_proba > 0.5).astype(int)
        val_acc = accuracy_score(y_val, val_pred)
        
        # Calculate metrics
        fold_metrics = calculate_metrics(y_val, val_pred)
        
        model_train_time = time.time() - model_train_start
        
        return model, history, val_acc, val_pred, val_pred_proba, scaler, fold_metrics

    def cross_validate_ensemble(self, X_url_train, y_train,
                               epochs=15, batch_size=128):
        """
        Perform cross-validation with ensemble.
        
        Args:
            X_url_train: Training URLs
            y_train: Training labels
            epochs: Training epochs
            batch_size: Batch size
        
        Returns:
            CV scores per model and ensemble scores
        """
        print(f"\n🔬 {self.n_folds}-Fold Cross-Validation Start...")
        
        cv_start_time = time.time()
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        model_types = ['base', 'multi_cnn', 'attention', 'wide'][:self.n_models]
        
        cv_scores_per_model = {model_type: [] for model_type in model_types}
        cv_ensemble_scores = []
        
        cv_fold_confusion_matrices = []
        cv_fold_metrics = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_url_train, y_train)):
            print(f"\n{'='*60}")
            print(f"📊 FOLD {fold_idx+1}/{self.n_folds}")
            print(f"{'='*60}")
            
            # Split data
            X_url_fold_train = X_url_train[train_idx]
            X_url_fold_val = X_url_train[val_idx]
            y_fold_train = y_train[train_idx]
            y_fold_val = y_train[val_idx]
            
            # Create isolated features for this fold
            train_data = list(zip(X_url_fold_train, y_fold_train))
            val_data = list(zip(X_url_fold_val, y_fold_val))
            
            train_cache = self.feature_manager.create_features_for_data(
                train_data, f"fold_{fold_idx+1}_train"
            )
            val_cache = self.feature_manager.create_features_for_data(
                val_data, f"fold_{fold_idx+1}_val"
            )
            
            # Extract features
            X_num_fold_train, y_fold_train_extracted, X_url_fold_train_processed = \
                extract_features_for_fold_optimized(
                    X_url_fold_train, y_fold_train, is_train=True,
                    bow_data=train_cache['bow'],
                    seg_bow_data=train_cache['seg_bow'],
                    ngrams_data=train_cache['ngrams'],
                    grams4_data=train_cache['grams4'],
                    tld_data=train_cache['tld']
                )
            
            X_num_fold_val, y_fold_val_extracted, X_url_fold_val_processed = \
                extract_features_for_fold_optimized(
                    X_url_fold_val, y_fold_val, is_train=False,
                    bow_data=val_cache['bow'],
                    seg_bow_data=val_cache['seg_bow'],
                    ngrams_data=val_cache['ngrams'],
                    grams4_data=val_cache['grams4'],
                    tld_data=val_cache['tld']
                )
            
            # Calculate class weights
            classes_fold = np.unique(y_fold_train_extracted)
            weights_fold = compute_class_weight(
                class_weight='balanced',
                classes=classes_fold,
                y=y_fold_train_extracted
            )
            class_weight_dict = {cls: w for cls, w in zip(classes_fold, weights_fold)}
            
            # Train models for this fold
            fold_predictions = []
            fold_probabilities = []
            
            for model_idx, (model_type, seed) in enumerate(zip(model_types, self.random_seeds)):
                print(f"\n🔧 Model {model_idx+1}: {model_type}")
                
                model, history, val_acc, val_pred, val_pred_proba, scaler, metrics = \
                    self.train_model_fold(
                        X_url_fold_train_processed, X_num_fold_train, y_fold_train_extracted,
                        X_url_fold_val_processed, X_num_fold_val, y_fold_val_extracted,
                        model_type, seed, fold_idx+1, class_weight_dict,
                        epochs, batch_size
                    )
                
                fold_predictions.append(val_pred)
                fold_probabilities.append(val_pred_proba)
                cv_scores_per_model[model_type].append(val_acc)
                
                print(f"   ✓ Accuracy: {val_acc:.4f}")
            
            # Ensemble prediction
            ensemble_pred_proba = np.mean(fold_probabilities, axis=0)
            ensemble_pred = (ensemble_pred_proba > 0.5).astype(int)
            ensemble_acc = accuracy_score(y_fold_val_extracted, ensemble_pred)
            cv_ensemble_scores.append(ensemble_acc)
            
            ensemble_metrics = calculate_metrics(y_fold_val_extracted, ensemble_pred)
            cv_fold_confusion_matrices.append(ensemble_metrics['confusion_matrix'])
            cv_fold_metrics.append(ensemble_metrics)
            
            print(f"\n📊 Ensemble Accuracy: {ensemble_acc:.4f}")
            
            # Cleanup
            self.cleanup_gpu_memory(fold_idx=fold_idx)
        
        # Calculate averages
        self.cv_confusion_matrices = cv_fold_confusion_matrices
        self.cv_metrics = cv_fold_metrics
        self.avg_confusion_matrix = np.mean(cv_fold_confusion_matrices, axis=0)
        
        # Calculate average metrics
        self.avg_metrics = {}
        metric_keys = ['accuracy', 'precision', 'recall', 'f1_score', 
                      'sensitivity', 'specificity', 'fpr', 'fnr']
        
        for key in metric_keys:
            values = [m[key] for m in cv_fold_metrics]
            self.avg_metrics[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        
        self.cv_time = time.time() - cv_start_time
        self.cv_scores = {
            'individual': cv_scores_per_model,
            'ensemble': cv_ensemble_scores
        }
        
        print(f"\n✓ Cross-Validation Complete!")
        print(f"⏱️  Total Time: {self.format_time(self.cv_time)}")
        
        return cv_scores_per_model, cv_ensemble_scores

    def train_final_ensemble(self, X_url_train, y_train, X_url_test, y_test,
                            epochs=15, batch_size=128):
        """
        Train final ensemble on all training data.
        
        Args:
            X_url_train: Training URLs
            y_train: Training labels
            X_url_test: Test URLs
            y_test: Test labels
            epochs: Training epochs
            batch_size: Batch size
        """
        print(f"\n🎯 Final Ensemble Training...")
        
        training_start = time.time()
        
        # Create features
        train_data = list(zip(X_url_train, y_train))
        test_data = list(zip(X_url_test, y_test))
        
        train_cache = self.feature_manager.create_features_for_data(
            train_data, "final_train"
        )
        test_cache = self.feature_manager.create_features_for_data(
            test_data, "final_test"
        )
        
        # Extract features
        X_num_train, y_train_extracted, X_url_train_processed = \
            extract_features_for_fold_optimized(
                X_url_train, y_train, is_train=True,
                bow_data=train_cache['bow'],
                seg_bow_data=train_cache['seg_bow'],
                ngrams_data=train_cache['ngrams'],
                grams4_data=train_cache['grams4'],
                tld_data=train_cache['tld']
            )
        
        X_num_test, y_test_extracted, X_url_test_processed = \
            extract_features_for_fold_optimized(
                X_url_test, y_test, is_train=False,
                bow_data=test_cache['bow'],
                seg_bow_data=test_cache['seg_bow'],
                ngrams_data=test_cache['ngrams'],
                grams4_data=test_cache['grams4'],
                tld_data=test_cache['tld']
            )
        
        # Scale features
        self.scaler = StandardScaler()
        X_num_train_scaled = self.scaler.fit_transform(X_num_train)
        X_num_test_scaled = self.scaler.transform(X_num_test)
        
        # Tokenize
        seq_train = self.tokenizer.texts_to_sequences(X_url_train_processed)
        seq_test = self.tokenizer.texts_to_sequences(X_url_test_processed)
        
        X_url_train_pad = pad_sequences(seq_train, maxlen=self.max_len,
                                       padding="post", truncating="post")
        X_url_test_pad = pad_sequences(seq_test, maxlen=self.max_len,
                                      padding="post", truncating="post")
        
        # Calculate class weights
        classes = np.unique(y_train_extracted)
        weights = compute_class_weight(
            class_weight='balanced', classes=classes, y=y_train_extracted
        )
        class_weight_dict = {cls: w for cls, w in zip(classes, weights)}
        
        # Train each model
        model_types = ['base', 'multi_cnn', 'attention', 'wide'][:self.n_models]
        
        for i, (model_type, seed) in enumerate(zip(model_types, self.random_seeds)):
            print(f"\n{'='*70}")
            print(f"FINAL MODEL {i+1}/{self.n_models}: {model_type.upper()}")
            print(f"{'='*70}")
            
            model = create_model_architecture(
                self.vocab_size, self.max_len, X_num_train_scaled.shape[1],
                model_type=model_type, seed=seed
            )
            model = compile_model(model)
            
            callbacks = self.create_optimized_callbacks(f"final_model_{i+1}")
            
            history = model.fit(
                x={"url_input": X_url_train_pad, "num_input": X_num_train_scaled},
                y=y_train_extracted,
                validation_data=(
                    {"url_input": X_url_test_pad, "num_input": X_num_test_scaled},
                    y_test_extracted
                ),
                epochs=epochs,
                batch_size=batch_size,
                class_weight=class_weight_dict,
                callbacks=callbacks,
                verbose=1
            )
            
            # Evaluate
            test_loss, test_acc = model.evaluate(
                {"url_input": X_url_test_pad, "num_input": X_num_test_scaled},
                y_test_extracted, verbose=0
            )
            
            print(f"\n✓ Test Accuracy: {test_acc:.4f}")
            
            self.models.append(model)
            self.histories.append(history)
            self.model_info.append({
                'type': model_type,
                'test_accuracy': test_acc,
                'loss': test_loss
            })
        
        self.total_training_time = time.time() - training_start
        self.cleanup_gpu_memory()
        
        print(f"\n✓ Final training completed in {self.format_time(self.total_training_time)}")

    def predict_ensemble(self, X_url, X_num, method='soft_voting'):
        """
        Make ensemble predictions.
        
        Args:
            X_url: URLs to predict
            X_num: Numerical features
            method: Voting method ('soft_voting', 'hard_voting', 'weighted_voting')
        
        Returns:
            Predictions and probabilities
        """
        # Scale and tokenize
        X_num_scaled = self.scaler.transform(X_num)
        seq_data = self.tokenizer.texts_to_sequences(X_url)
        X_url_pad = pad_sequences(seq_data, maxlen=self.max_len,
                                 padding="post", truncating="post")
        
        # Get predictions from all models
        predictions = []
        for model in self.models:
            pred = model.predict(
                {"url_input": X_url_pad, "num_input": X_num_scaled},
                verbose=0
            )
            predictions.append(pred.flatten())
        
        predictions = np.array(predictions)
        
        if method == 'soft_voting':
            ensemble_pred_proba = np.mean(predictions, axis=0)
            ensemble_pred = (ensemble_pred_proba > 0.5).astype(int)
        elif method == 'hard_voting':
            hard_preds = (predictions > 0.5).astype(int)
            ensemble_pred = np.round(np.mean(hard_preds, axis=0)).astype(int)
            ensemble_pred_proba = np.mean(predictions, axis=0)
        else:  # weighted_voting
            weights = np.array([info['test_accuracy'] for info in self.model_info])
            weights = weights / np.sum(weights)
            ensemble_pred_proba = np.average(predictions, axis=0, weights=weights)
            ensemble_pred = (ensemble_pred_proba > 0.5).astype(int)
        
        return ensemble_pred, ensemble_pred_proba
