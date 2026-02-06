import os
import json
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from .config import MODELS_DIR, CHECKPOINT_DIR, RESULTS_DIR

class CheckpointManager:
    """
    Manages training checkpoints for Cross-Validation to allow
    resuming from the last completed fold.
    """
    def __init__(self, checkpoint_dir=CHECKPOINT_DIR):
        self.checkpoint_dir = checkpoint_dir
        self.metadata_file = os.path.join(self.checkpoint_dir, "metadata.json")
        
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            print(f"✓ Checkpoint directory created: {self.checkpoint_dir}")

    def save_fold_checkpoint(self, fold_idx, fold_models, fold_histories, 
                            fold_result, fold_metrics, model_metrics_list=None, 
                            timing_data=None):
        checkpoint_name = f"fold_{fold_idx+1}"
        fold_dir = os.path.join(self.checkpoint_dir, checkpoint_name)
        os.makedirs(fold_dir, exist_ok=True)

        try:
            # Save Keras models
            models_dir = os.path.join(fold_dir, "models")
            os.makedirs(models_dir, exist_ok=True)
            for i, model in enumerate(fold_models):
                model.save(os.path.join(models_dir, f"model_{i+1}.keras"))

            # Save Metrics and Histories
            with open(os.path.join(fold_dir, "histories.pkl"), "wb") as f:
                pickle.dump(fold_histories, f)
            with open(os.path.join(fold_dir, "fold_result.pkl"), "wb") as f:
                pickle.dump(fold_result, f)
            with open(os.path.join(fold_dir, "fold_metrics.pkl"), "wb") as f:
                pickle.dump(fold_metrics, f)

            if model_metrics_list:
                with open(os.path.join(fold_dir, "model_metrics_list.pkl"), "wb") as f:
                    pickle.dump(model_metrics_list, f)
            if timing_data:
                with open(os.path.join(fold_dir, "timing_data.pkl"), "wb") as f:
                    pickle.dump(timing_data, f)

            self._update_metadata(fold_idx)
            print(f"✓ Fold {fold_idx+1} checkpoint saved successfully.")
        except Exception as e:
            print(f"✗ Checkpoint save error: {e}")

    def load_fold_checkpoint(self, fold_idx):
        fold_dir = os.path.join(self.checkpoint_dir, f"fold_{fold_idx+1}")
        if not os.path.exists(fold_dir):
            return None

        try:
            models_dir = os.path.join(fold_dir, "models")
            fold_models = []
            for i in range(1, 5): # Assuming max 4 models
                m_path = os.path.join(models_dir, f"model_{i}.keras")
                if os.path.exists(m_path):
                    fold_models.append(load_model(m_path))

            data = {
                'models': fold_models,
                'histories': pickle.load(open(os.path.join(fold_dir, "histories.pkl"), "rb")),
                'fold_result': pickle.load(open(os.path.join(fold_dir, "fold_result.pkl"), "rb")),
                'fold_metrics': pickle.load(open(os.path.join(fold_dir, "fold_metrics.pkl"), "rb"))
            }
            
            # Optional files
            for opt in ["model_metrics_list", "timing_data"]:
                p = os.path.join(fold_dir, f"{opt}.pkl")
                if os.path.exists(p):
                    data[opt] = pickle.load(open(p, "rb"))

            return data
        except Exception:
            return None

    def get_completed_folds(self):
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f).get('completed_folds', [])
            except: pass
        return []

    def get_last_completed_fold(self):
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f).get('last_completed_fold', -1)
            except: pass
        return -1

    def _update_metadata(self, fold_idx):
        metadata = {'completed_folds': [], 'last_completed_fold': -1}
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
        
        if fold_idx not in metadata['completed_folds']:
            metadata['completed_folds'].append(fold_idx)
        metadata['last_completed_fold'] = fold_idx
        
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

def create_model_architecture(vocab_size, max_len, n_features, model_type='base', seed=42):
    """
    Factory function to create different deep learning architectures
    for URL phishing detection.
    """
    tf.random.set_seed(seed)
    np.random.seed(seed)

    url_input = Input(shape=(max_len,), name="url_input")
    num_input = Input(shape=(n_features,), name="num_input")

    # URL Text Branch (Embedding)
    embedding_dim = 64
    embed_layer = Embedding(vocab_size, embedding_dim, input_length=max_len, mask_zero=True)(url_input)

    if model_type == 'base':
        conv1 = Conv1D(64, 3, activation="relu", padding="same", kernel_regularizer=l2(0.001))(embed_layer)
        conv1 = BatchNormalization()(conv1)
        pool1 = GlobalMaxPooling1D()(conv1)
        lstm_out = Bidirectional(LSTM(32, dropout=0.3, recurrent_dropout=0.3))(embed_layer)
        url_features = Concatenate()([pool1, lstm_out])

    elif model_type == 'multi_cnn':
        conv_3 = Conv1D(32, 3, activation='relu', padding='same')(embed_layer)
        conv_5 = Conv1D(32, 5, activation='relu', padding='same')(embed_layer)
        pool_3 = GlobalMaxPooling1D()(conv_3)
        pool_5 = GlobalMaxPooling1D()(conv_5)
        lstm_out = Bidirectional(LSTM(32, dropout=0.3))(embed_layer)
        url_features = Concatenate()([pool_3, pool_5, lstm_out])

    elif model_type == 'attention':
        attention_layer = Dense(embedding_dim, activation='tanh')(embed_layer)
        attention_weights = Dense(1, activation='softmax')(attention_layer)
        attention_out = Multiply()([embed_layer, attention_weights])
        attention_pooled = GlobalAveragePooling1D()(attention_out)
        conv1 = Conv1D(32, 3, activation="relu", padding="same")(embed_layer)
        pool1 = GlobalMaxPooling1D()(conv1)
        url_features = Concatenate()([attention_pooled, pool1])

    else:  # 'wide' model
        conv1 = Conv1D(64, 3, activation="relu", padding="same")(embed_layer)
        pool1 = GlobalMaxPooling1D()(conv1)
        lstm_out = Bidirectional(LSTM(64, dropout=0.3))(embed_layer)
        url_features = Concatenate()([pool1, lstm_out])

    # Numerical Features Branch
    num_dense = Dense(32, activation="relu", kernel_regularizer=l2(0.001))(num_input)
    num_batch = BatchNormalization()(num_dense)
    num_drop = Dropout(0.3)(num_batch)

    # Fusion
    merged = Concatenate()([url_features, num_drop])

    # Final Dense Layers
    dense1 = Dense(64, activation="relu", kernel_regularizer=l2(0.001))(merged)
    batch1 = BatchNormalization()(dense1)
    drop1 = Dropout(0.4)(batch1)

    dense2 = Dense(32, activation="relu", kernel_regularizer=l2(0.001))(drop1)
    batch2 = BatchNormalization()(dense2)
    drop2 = Dropout(0.3)(batch2)

    output = Dense(1, activation="sigmoid", dtype='float32')(drop2)
    
    return Model(inputs=[url_input, num_input], outputs=output)

def get_callbacks(model_name):
    """Returns optimized training callbacks."""
    return [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, min_lr=1e-7, verbose=1)
    ]