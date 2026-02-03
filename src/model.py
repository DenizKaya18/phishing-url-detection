"""
Deep learning model architectures for URL phishing detection.
Contains ensemble of CNN-BiLSTM based models with different architectures.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Embedding, Conv1D, BatchNormalization, GlobalMaxPooling1D,
    GlobalAveragePooling1D, Bidirectional, LSTM, Dense, Dropout, 
    Concatenate, Multiply
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam


class ModelArchitecture:
    """
    Factory class for creating different model architectures.
    """
    
    @staticmethod
    def create_base_model(vocab_size, max_len, n_features, seed=42):
        """
        Base model: CNN + BiLSTM with regularization.
        
        Args:
            vocab_size: Size of vocabulary
            max_len: Maximum sequence length
            n_features: Number of numerical features
            seed: Random seed for reproducibility
        
        Returns:
            Compiled Keras model
        """
        tf.random.set_seed(seed)
        np.random.seed(seed)
        
        # URL input branch
        url_input = Input(shape=(max_len,), name="url_input")
        
        embedding_dim = 64
        embed_layer = Embedding(
            vocab_size, embedding_dim,
            input_length=max_len, mask_zero=True
        )(url_input)
        
        # CNN branch
        conv1 = Conv1D(
            64, 3, activation="relu", padding="same",
            kernel_regularizer=l2(0.001)
        )(embed_layer)
        conv1 = BatchNormalization()(conv1)
        pool1 = GlobalMaxPooling1D()(conv1)
        
        # BiLSTM branch
        lstm_out = Bidirectional(LSTM(
            32, return_sequences=False,
            kernel_regularizer=l2(0.001),
            recurrent_regularizer=l2(0.001),
            dropout=0.3, recurrent_dropout=0.3
        ))(embed_layer)
        
        # Concatenate URL features
        url_features = Concatenate()([pool1, lstm_out])
        
        # Numerical features branch
        num_input = Input(shape=(n_features,), name="num_input")
        num_dense = Dense(32, activation="relu", kernel_regularizer=l2(0.001))(num_input)
        num_batch = BatchNormalization()(num_dense)
        num_drop = Dropout(0.3)(num_batch)
        
        # Merge branches
        merged = Concatenate()([url_features, num_drop])
        
        # Classification head
        dense1 = Dense(64, activation="relu", kernel_regularizer=l2(0.001))(merged)
        batch1 = BatchNormalization()(dense1)
        drop1 = Dropout(0.4)(batch1)
        
        dense2 = Dense(32, activation="relu", kernel_regularizer=l2(0.001))(drop1)
        batch2 = BatchNormalization()(dense2)
        drop2 = Dropout(0.3)(batch2)
        
        output = Dense(1, activation="sigmoid", dtype='float32')(drop2)
        
        model = Model(inputs=[url_input, num_input], outputs=output)
        return model
    
    @staticmethod
    def create_multi_cnn_model(vocab_size, max_len, n_features, seed=42):
        """
        Multi-scale CNN model with different kernel sizes.
        
        Args:
            vocab_size: Size of vocabulary
            max_len: Maximum sequence length
            n_features: Number of numerical features
            seed: Random seed
        
        Returns:
            Compiled Keras model
        """
        tf.random.set_seed(seed)
        np.random.seed(seed)
        
        url_input = Input(shape=(max_len,), name="url_input")
        
        embedding_dim = 64
        embed_layer = Embedding(
            vocab_size, embedding_dim,
            input_length=max_len, mask_zero=True
        )(url_input)
        
        # Multiple CNN branches with different kernel sizes
        conv_3 = Conv1D(32, 3, activation='relu', padding='same')(embed_layer)
        conv_5 = Conv1D(32, 5, activation='relu', padding='same')(embed_layer)
        
        pool_3 = GlobalMaxPooling1D()(conv_3)
        pool_5 = GlobalMaxPooling1D()(conv_5)
        
        # BiLSTM branch
        lstm_out = Bidirectional(LSTM(
            32, return_sequences=False, dropout=0.3
        ))(embed_layer)
        
        url_features = Concatenate()([pool_3, pool_5, lstm_out])
        
        # Numerical features
        num_input = Input(shape=(n_features,), name="num_input")
        num_dense = Dense(32, activation="relu", kernel_regularizer=l2(0.001))(num_input)
        num_batch = BatchNormalization()(num_dense)
        num_drop = Dropout(0.3)(num_batch)
        
        merged = Concatenate()([url_features, num_drop])
        
        dense1 = Dense(64, activation="relu", kernel_regularizer=l2(0.001))(merged)
        batch1 = BatchNormalization()(dense1)
        drop1 = Dropout(0.4)(batch1)
        
        dense2 = Dense(32, activation="relu", kernel_regularizer=l2(0.001))(drop1)
        batch2 = BatchNormalization()(dense2)
        drop2 = Dropout(0.3)(batch2)
        
        output = Dense(1, activation="sigmoid", dtype='float32')(drop2)
        
        model = Model(inputs=[url_input, num_input], outputs=output)
        return model
    
    @staticmethod
    def create_attention_model(vocab_size, max_len, n_features, seed=42):
        """
        Model with attention mechanism.
        
        Args:
            vocab_size: Size of vocabulary
            max_len: Maximum sequence length
            n_features: Number of numerical features
            seed: Random seed
        
        Returns:
            Compiled Keras model
        """
        tf.random.set_seed(seed)
        np.random.seed(seed)
        
        url_input = Input(shape=(max_len,), name="url_input")
        
        embedding_dim = 64
        embed_layer = Embedding(
            vocab_size, embedding_dim,
            input_length=max_len, mask_zero=True
        )(url_input)
        
        # Attention mechanism
        attention_layer = Dense(embedding_dim, activation='tanh')(embed_layer)
        attention_weights = Dense(1, activation='softmax')(attention_layer)
        attention_out = Multiply()([embed_layer, attention_weights])
        attention_pooled = GlobalAveragePooling1D()(attention_out)
        
        # CNN branch
        conv1 = Conv1D(32, 3, activation="relu", padding="same")(embed_layer)
        pool1 = GlobalMaxPooling1D()(conv1)
        
        url_features = Concatenate()([attention_pooled, pool1])
        
        # Numerical features
        num_input = Input(shape=(n_features,), name="num_input")
        num_dense = Dense(32, activation="relu", kernel_regularizer=l2(0.001))(num_input)
        num_batch = BatchNormalization()(num_dense)
        num_drop = Dropout(0.3)(num_batch)
        
        merged = Concatenate()([url_features, num_drop])
        
        dense1 = Dense(64, activation="relu", kernel_regularizer=l2(0.001))(merged)
        batch1 = BatchNormalization()(dense1)
        drop1 = Dropout(0.4)(batch1)
        
        dense2 = Dense(32, activation="relu", kernel_regularizer=l2(0.001))(drop1)
        batch2 = BatchNormalization()(dense2)
        drop2 = Dropout(0.3)(batch2)
        
        output = Dense(1, activation="sigmoid", dtype='float32')(drop2)
        
        model = Model(inputs=[url_input, num_input], outputs=output)
        return model
    
    @staticmethod
    def create_wide_model(vocab_size, max_len, n_features, seed=42):
        """
        Wide model with increased capacity.
        
        Args:
            vocab_size: Size of vocabulary
            max_len: Maximum sequence length
            n_features: Number of numerical features
            seed: Random seed
        
        Returns:
            Compiled Keras model
        """
        tf.random.set_seed(seed)
        np.random.seed(seed)
        
        url_input = Input(shape=(max_len,), name="url_input")
        
        embedding_dim = 64
        embed_layer = Embedding(
            vocab_size, embedding_dim,
            input_length=max_len, mask_zero=True
        )(url_input)
        
        # Wider CNN
        conv1 = Conv1D(64, 3, activation="relu", padding="same")(embed_layer)
        conv1 = BatchNormalization()(conv1)
        pool1 = GlobalMaxPooling1D()(conv1)
        
        # Wider BiLSTM
        lstm_out = Bidirectional(LSTM(
            64, return_sequences=False, dropout=0.3
        ))(embed_layer)
        
        url_features = Concatenate()([pool1, lstm_out])
        
        # Numerical features
        num_input = Input(shape=(n_features,), name="num_input")
        num_dense = Dense(32, activation="relu", kernel_regularizer=l2(0.001))(num_input)
        num_batch = BatchNormalization()(num_dense)
        num_drop = Dropout(0.3)(num_batch)
        
        merged = Concatenate()([url_features, num_drop])
        
        dense1 = Dense(64, activation="relu", kernel_regularizer=l2(0.001))(merged)
        batch1 = BatchNormalization()(dense1)
        drop1 = Dropout(0.4)(batch1)
        
        dense2 = Dense(32, activation="relu", kernel_regularizer=l2(0.001))(drop1)
        batch2 = BatchNormalization()(dense2)
        drop2 = Dropout(0.3)(batch2)
        
        output = Dense(1, activation="sigmoid", dtype='float32')(drop2)
        
        model = Model(inputs=[url_input, num_input], outputs=output)
        return model


def create_model_architecture(vocab_size, max_len, n_features, 
                              model_type='base', seed=42):
    """
    Factory function to create model based on type.
    
    Args:
        vocab_size: Size of vocabulary
        max_len: Maximum sequence length
        n_features: Number of numerical features
        model_type: Type of model ('base', 'multi_cnn', 'attention', 'wide')
        seed: Random seed for reproducibility
    
    Returns:
        Uncompiled Keras model
    """
    model_factory = {
        'base': ModelArchitecture.create_base_model,
        'multi_cnn': ModelArchitecture.create_multi_cnn_model,
        'attention': ModelArchitecture.create_attention_model,
        'wide': ModelArchitecture.create_wide_model
    }
    
    if model_type not in model_factory:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model_factory[model_type](vocab_size, max_len, n_features, seed)


def compile_model(model, learning_rate=0.008):
    """
    Compile model with optimizer and loss function.
    
    Args:
        model: Keras model to compile
        learning_rate: Learning rate for optimizer
    
    Returns:
        Compiled model
    """
    base_opt = Adam(
        learning_rate=learning_rate,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7
    )
    
    # Use mixed precision optimizer if available
    try:
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(base_opt, dynamic=True)
    except:
        optimizer = base_opt
    
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    
    return model
