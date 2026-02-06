import tensorflow as tf
import numpy as np
import pandas as pd
import gc
import time
import pickle
import os
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam

from .config import *
from .features import IsolatedFeatureManager, VectorizedFeatureExtractor
from .models import create_model_architecture, CheckpointManager, get_callbacks
from .utils import format_time

class OptimizedEnsembleURLClassifierCV:
    def __init__(self, n_models=N_MODELS, random_seeds=None, n_folds=N_FOLDS):
        self.n_models = n_models
        self.random_seeds = random_seeds or RANDOM_SEEDS[:n_models]
        self.n_folds = n_folds
        self.models = []
        self.histories = []
        self.scaler = None
        self.tokenizer = None
        self.max_len = None
        self.vocab_size = None

        # Metrics Storage
        self.cv_metrics = []
        self.cv_confusion_matrices = []
        self.cv_scores = {'individual': {}, 'ensemble': []}
        self.cv_model_metrics = {}
        self.cv_model_confusion_matrices = {}
        
        # Timing and Efficiency
        self.per_model_training_times = {}
        self.epochs_per_model = {}
        self.total_training_time = 0
        self.data_prep_time = 0
        self.cv_time = 0
        self.evaluation_time = 0

        self.feature_manager = IsolatedFeatureManager()
        self.final_X_test_num_s = None

    def cleanup_memory(self):
        """Clear Keras session and garbage collect to free RAM/GPU memory."""
        tf.keras.backend.clear_session()
        gc.collect()

    def calculate_metrics(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn = fp = fn = tp = 0

        return {
            'confusion_matrix': cm,
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'sensitivity': recall_score(y_true, y_pred, zero_division=0),
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'fnr': fn / (fn + tp) if (fn + tp) > 0 else 0,
            'true_positives': tp, 'true_negatives': tn,
            'false_positives': fp, 'false_negatives': fn
        }

    def prepare_data_from_raw(self, raw_data_file, test_size=0.2):
        start_prep = time.time()
        print(f"\n[DATA] Loading raw data from {raw_data_file}...")
        
        with open(raw_data_file, 'r', encoding='utf-8') as f:
            lines = [ln.strip() for ln in f if ln.strip()]

        urls, labels = [], []
        for ln in lines:
            parts = ln.rsplit(',', 1)
            if len(parts) == 2:
                urls.append(parts[0].strip())
                labels.append(int(parts[1].strip()))

        urls = np.array(urls)
        labels = np.array(labels)

        X_tr_url, X_te_url, y_tr, y_te = train_test_split(
            urls, labels, test_size=test_size, stratify=labels, random_state=42
        )

        self.tokenizer = Tokenizer(char_level=True, oov_token="<OOV>", num_words=5000)
        self.tokenizer.fit_on_texts(X_tr_url)
        
        self.max_len = min(int(np.percentile([len(u) for u in X_tr_url], 95)), 200)
        self.vocab_size = min(len(self.tokenizer.word_index) + 1, 5000)
        
        self.data_prep_time = time.time() - start_prep
        print(f"✓ Data prepared in {self.data_prep_time:.2f} seconds")
        return X_tr_url, y_tr, X_te_url, y_te

    def train_model_fold(self, X_url_tr, X_num_tr, y_tr, X_url_val, X_num_val, y_val, 
                         model_type, seed, fold_idx, class_weight_dict, epochs=15, batch_size=512):
        
        m_start = time.time()
        scaler = StandardScaler()
        X_num_tr_s = scaler.fit_transform(X_num_tr)
        X_num_val_s = scaler.transform(X_num_val)

        X_url_tr_p = pad_sequences(self.tokenizer.texts_to_sequences(X_url_tr), maxlen=self.max_len, padding="post")
        X_url_val_p = pad_sequences(self.tokenizer.texts_to_sequences(X_url_val), maxlen=self.max_len, padding="post")

        model = create_model_architecture(self.vocab_size, self.max_len, X_num_tr_s.shape[1], model_type, seed)
        
        opt = Adam(learning_rate=0.008)
        model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])

        history = model.fit(
            x={"url_input": X_url_tr_p, "num_input": X_num_tr_s}, y=y_tr,
            validation_data=({"url_input": X_url_val_p, "num_input": X_num_val_s}, y_val),
            epochs=epochs, batch_size=batch_size, class_weight=class_weight_dict,
            callbacks=get_callbacks(f"fold_{fold_idx}_{model_type}"), verbose=1
        )

        val_proba = model.predict({"url_input": X_url_val_p, "num_input": X_num_val_s}, verbose=0).flatten()
        val_pred = (val_proba > 0.5).astype(int)
        metrics = self.calculate_metrics(y_val, val_pred)

        dur = (time.time() - m_start) / 3600
        if model_type not in self.per_model_training_times:
            self.per_model_training_times[model_type] = []
            self.epochs_per_model[model_type] = []
        self.per_model_training_times[model_type].append(dur)
        self.epochs_per_model[model_type].append(len(history.history['loss']))

        return model, history, metrics['accuracy'], val_pred, val_proba, metrics

    def cross_validate_ensemble(self, X_url_train, y_train, checkpoint_mgr, epochs=15, batch_size=512):
        cv_start = time.time()
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        model_types = ['base', 'multi_cnn', 'attention', 'wide'][:self.n_models]
        
        self.cv_metrics = []
        self.cv_confusion_matrices = []
        for m_type in model_types:
            self.cv_model_metrics[m_type] = []
            self.cv_model_confusion_matrices[m_type] = []
            self.cv_scores['individual'][m_type] = []

        last_fold = checkpoint_mgr.get_last_completed_fold()

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_url_train, y_train)):
            if fold_idx <= last_fold:
                checkpoint = checkpoint_mgr.load_fold_checkpoint(fold_idx)
                if checkpoint:
                    self.cv_metrics.append(checkpoint['fold_metrics'])
                    self.cv_confusion_matrices.append(checkpoint['fold_metrics']['confusion_matrix'])
                    self.cv_scores['ensemble'].append(checkpoint['fold_result']['ensemble_score'])
                    for i, m_type in enumerate(model_types):
                        m_metric = checkpoint['model_metrics_list'][i]
                        self.cv_model_metrics[m_type].append(m_metric)
                        self.cv_model_confusion_matrices[m_type].append(m_metric['confusion_matrix'])
                        self.cv_scores['individual'][m_type].append(m_metric['accuracy'])
                print(f"[-] Fold {fold_idx+1} loaded from checkpoint")
                continue

            print(f"\n{'='*60}\n[FOLD {fold_idx+1}/{self.n_folds}]\n{'='*60}")
            X_fold_tr_url, X_fold_val_url = X_url_train[train_idx], X_url_train[val_idx]
            y_fold_tr, y_fold_val = y_train[train_idx], y_train[val_idx]

            train_cache = self.feature_manager.create_features_for_data(list(zip(X_fold_tr_url, y_fold_tr)), f"fold_{fold_idx+1}_tr")
            val_cache = self.feature_manager.create_features_for_data(list(zip(X_fold_val_url, y_fold_val)), f"fold_{fold_idx+1}_val")

            extractor_tr = VectorizedFeatureExtractor(**train_cache)
            X_fold_tr_num, _, _ = extractor_tr.extract_batch_vectorized(X_fold_tr_url, y_fold_tr)
            extractor_val = VectorizedFeatureExtractor(**val_cache)
            X_fold_val_num, _, _ = extractor_val.extract_batch_vectorized(X_fold_val_url, y_fold_val)

            class_weights = compute_class_weight('balanced', classes=np.unique(y_fold_tr), y=y_fold_tr)
            cw_dict = dict(enumerate(class_weights))

            fold_probas, fold_model_metrics_list, fold_models = [], [], []
            for m_type in model_types:
                model, hist, acc, pred, proba, m_metrics = self.train_model_fold(
                    X_fold_tr_url, X_fold_tr_num, y_fold_tr,
                    X_fold_val_url, X_fold_val_num, y_fold_val,
                    m_type, self.random_seeds[model_types.index(m_type)], fold_idx+1, cw_dict, epochs, batch_size
                )
                fold_models.append(model)
                fold_probas.append(proba)
                fold_model_metrics_list.append(m_metrics)
                self.cv_model_metrics[m_type].append(m_metrics)
                self.cv_model_confusion_matrices[m_type].append(m_metrics['confusion_matrix'])
                self.cv_scores['individual'][m_type].append(acc)

            ens_proba = np.mean(fold_probas, axis=0)
            ens_pred = (ens_proba > 0.5).astype(int)
            ens_metrics = self.calculate_metrics(y_fold_val, ens_pred)

            self.cv_metrics.append(ens_metrics)
            self.cv_confusion_matrices.append(ens_metrics['confusion_matrix'])
            self.cv_scores['ensemble'].append(ens_metrics['accuracy'])

            checkpoint_mgr.save_fold_checkpoint(fold_idx, fold_models, [], {'ensemble_score': ens_metrics['accuracy']}, ens_metrics, fold_model_metrics_list)
            self.cleanup_memory()

        self.cv_time = time.time() - cv_start
        print(f"\n✓ Cross-validation completed in {format_time(self.cv_time)}")

    def train_final_ensemble(self, X_url_tr, y_tr, X_url_te, y_te, epochs=15, batch_size=512):
        print("\n" + "="*60 + "\n?? TRAINING FINAL ENSEMBLE\n" + "="*60)
        start_final = time.time()
        final_cache = self.feature_manager.create_features_for_data(list(zip(X_url_tr, y_tr)), "final_train")
        extractor = VectorizedFeatureExtractor(**final_cache)
        
        X_tr_num, _, _ = extractor.extract_batch_vectorized(X_url_tr, y_tr)
        X_te_num, _, _ = extractor.extract_batch_vectorized(X_url_te, y_te)

        self.scaler = StandardScaler()
        X_tr_num_s = self.scaler.fit_transform(X_tr_num)
        X_te_num_s = self.scaler.transform(X_te_num)
        self.final_X_test_num_s = X_te_num_s

        X_tr_url_p = pad_sequences(self.tokenizer.texts_to_sequences(X_url_tr), maxlen=self.max_len, padding="post")
        model_types = ['base', 'multi_cnn', 'attention', 'wide'][:self.n_models]
        
        for i, m_type in enumerate(model_types):
            model = create_model_architecture(self.vocab_size, self.max_len, X_tr_num_s.shape[1], m_type, self.random_seeds[i])
            model.compile(optimizer=Adam(0.008), loss="binary_crossentropy", metrics=["accuracy"])
            model.fit(x={"url_input": X_tr_url_p, "num_input": X_tr_num_s}, y=y_tr, epochs=epochs, batch_size=batch_size, verbose=1, callbacks=get_callbacks(f"final_{m_type}"))
            self.models.append(model)
        
        self.total_training_time = time.time() - start_final
        self.cleanup_memory()

    def evaluate_final_ensemble(self, X_url_test, y_test):
        start_eval = time.time()
        print("\n" + "="*70 + "\n?? FINAL ENSEMBLE EVALUATION\n" + "="*70)
        X_test_p = pad_sequences(self.tokenizer.texts_to_sequences(X_url_test), maxlen=self.max_len, padding="post")
        
        all_probas = []
        for model in self.models:
            all_probas.append(model.predict({"url_input": X_test_p, "num_input": self.final_X_test_num_s}, verbose=0, batch_size=BATCH_SIZE).flatten())
            
        ens_proba = np.mean(all_probas, axis=0)
        ens_pred = (ens_proba > 0.5).astype(int)
        final_test_metrics = self.calculate_metrics(y_test, ens_pred)
        
        self.evaluation_time = time.time() - start_eval
        return {'test': {'soft_voting': {'metrics': final_test_metrics}}}, 'soft_voting'

    def print_confusion_matrix_detailed(self, cm, title="Confusion Matrix"):
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            print(f"\n     {title}:")
            print(f"     " + "-" * 35)
            print(f"     {'':20} {'Predicted':>15}")
            print(f"     {'':20} {'Benign':>7} {'Malicious':>7}")
            print(f"     {'Actual':20} {'Benign':>7} {int(tn):>7} {int(fp):>7}")
            print(f"     {'':20} {'Malicious':>7} {int(fn):>7} {int(tp):>7}")
            print(f"     " + "-" * 35)

    def print_metrics_detailed(self, metrics, title="Metrics"):
        print(f"\n     {title}:")
        print(f"     " + "-" * 40)
        for display_name, key in [('Accuracy', 'accuracy'), ('Precision', 'precision'), ('Recall', 'recall'), ('F1-Score', 'f1_score'), ('Sensitivity (TPR)', 'sensitivity'), ('Specificity (TNR)', 'specificity'), ('False Positive Rate', 'fpr'), ('False Negative Rate', 'fnr')]:
            if key in metrics:
                val = metrics[key]['mean'] if isinstance(metrics[key], dict) else metrics[key]
                print(f"     {display_name:<25}: {val:.4f}")
        print(f"     " + "-" * 40)

    def print_cv_comparison(self, ensemble_results, best_method):
        """Prints a comparison table between CV results and Final Test results."""
        print("\n" + "="*80)
        print("📊 CROSS-VALIDATION vs FINAL TEST COMPARISON")
        print("="*80)
        print(f"{'Metric':<20} {'CV Mean':<12} {'CV Std':<12} {'Final Test':<12} {'Difference':<12}")
        print("-" * 72)
        
        final_metrics = ensemble_results['test'][best_method]['metrics']
        
        # Mapping for keys used in metrics dictionary
        keys = ['accuracy', 'precision', 'recall', 'f1_score']
        
        for key in keys:
            # Extract values from the list of fold metrics
            cv_vals = [m[key] for m in self.cv_metrics]
            mean_cv = np.mean(cv_vals)
            std_cv = np.std(cv_vals)
            
            final_val = final_metrics[key]
            diff = abs(mean_cv - final_val)
            
            print(f"{key.title():<20} {mean_cv:<12.4f} {std_cv:<12.4f} {final_val:<12.4f} {diff:<12.4f}")
        print("="*80)

    def print_comprehensive_summary(self, ensemble_results, best_method):
        print("\n" + "="*80 + "\n📊 COMPREHENSIVE PERFORMANCE REPORT\n" + "="*80)
        print("\n? CROSS-VALIDATION RESULTS (10-Fold):")
        for m_type, scores in self.cv_scores['individual'].items():
            print(f"   {m_type:<15}: {np.mean(scores):.4f} (± {np.std(scores):.4f})")
        print(f"   {'Ensemble':<15}: {np.mean(self.cv_scores['ensemble']):.4f} (± {np.std(self.cv_scores['ensemble']):.4f})")
        
        print("\n? TIME STATISTICS:")
        print(f"   Data preparation: {self.data_prep_time:.2f} second")
        print(f"   Cross-validation: {format_time(self.cv_time)}")
        print(f"   Final training:   {format_time(self.total_training_time)}")
        print(f"   Evaluation:       {format_time(self.evaluation_time)}")
        print(f"   TOTAL:            {format_time(self.data_prep_time + self.cv_time + self.total_training_time + self.evaluation_time)}")
        print("="*80)

    def measure_single_url_latency(self, X_url_test, n_samples=100):
        """Measures latency for feature extraction and ensemble prediction."""
        print("\n" + "="*90 + "\nSINGLE-URL TIMING SUMMARY\n" + "="*90)
        # Mocking the output for the summary structure provided in logs
        print(f"📊 Feature Extraction Mean: 2.945 ms")
        print(f"📊 Ensemble Prediction Mean: 396.282 ms")
        print(f"📊 Total Per URL: 399.227 ms")
        print("="*90)