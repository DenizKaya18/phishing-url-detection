"""
Data preprocessing module for URL phishing detection.
Handles data loading, tokenization, padding, and train-test splitting.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def load_raw_data(filepath):
    """
    Load raw URL dataset from text file.
    
    Args:
        filepath: Path to the raw data file (format: url,label)
    
    Returns:
        urls: numpy array of URLs
        labels: numpy array of labels (0=benign, 1=malicious)
    """
    with open(filepath, 'r', encoding='utf-8') as fr:
        raw_lines = [ln.strip() for ln in fr if ln.strip()]
    
    rows = []
    for ln in raw_lines:
        parts = ln.rsplit(',', 1)
        if len(parts) != 2:
            continue
        rows.append((parts[0].strip(), parts[1].strip()))
    
    urls = np.array([r[0] for r in rows])
    labels = np.array([int(r[1]) for r in rows])
    
    return urls, labels


def prepare_data_from_raw(raw_data_file, test_size=0.2, random_state=42):
    """
    Prepare data from raw file: load, split, and create initial tokenizer.
    
    Args:
        raw_data_file: Path to raw data file
        test_size: Fraction of data for testing (default: 0.2)
        random_state: Random seed for reproducibility
    
    Returns:
        X_url_train: Training URLs
        X_url_test: Test URLs
        y_train: Training labels
        y_test: Test labels
        tokenizer: Fitted tokenizer object
        max_len: Maximum sequence length
        vocab_size: Vocabulary size
    """
    # Load data
    urls, labels = load_raw_data(raw_data_file)
    
    # Stratified train-test split
    X_url_train, X_url_test, y_train, y_test = train_test_split(
        urls, labels, 
        test_size=test_size, 
        stratify=labels, 
        random_state=random_state
    )
    
    # Create and fit tokenizer on training data only
    tokenizer = Tokenizer(char_level=True, oov_token="<OOV>", num_words=5000)
    tokenizer.fit_on_texts(X_url_train)
    
    # Calculate max_len based on 95th percentile of training URL lengths
    url_lengths = [len(u) for u in X_url_train]
    max_len = min(int(np.percentile(url_lengths, 95)), 200)
    vocab_size = min(len(tokenizer.word_index) + 1, 5000)
    
    print(f"✓ Data loaded successfully")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Max sequence length: {max_len}")
    print(f"  Training samples: {len(X_url_train)}")
    print(f"  Test samples: {len(X_url_test)}")
    print(f"  Training class distribution: {np.bincount(y_train)}")
    print(f"  Test class distribution: {np.bincount(y_test)}")
    
    return X_url_train, X_url_test, y_train, y_test, tokenizer, max_len, vocab_size


def tokenize_and_pad_urls(urls, tokenizer, max_len):
    """
    Tokenize and pad URL sequences.
    
    Args:
        urls: Array of URL strings
        tokenizer: Fitted Tokenizer object
        max_len: Maximum sequence length for padding
    
    Returns:
        Padded sequences as numpy array
    """
    sequences = tokenizer.texts_to_sequences(urls)
    padded = pad_sequences(
        sequences, 
        maxlen=max_len, 
        padding="post", 
        truncating="post"
    )
    return padded


def scale_numerical_features(X_train, X_test=None, scaler=None):
    """
    Scale numerical features using StandardScaler.
    
    Args:
        X_train: Training feature matrix
        X_test: Test feature matrix (optional)
        scaler: Pre-fitted scaler (optional, for inference)
    
    Returns:
        X_train_scaled: Scaled training features
        X_test_scaled: Scaled test features (if X_test provided)
        scaler: Fitted scaler object
    """
    if scaler is None:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
    else:
        X_train_scaled = scaler.transform(X_train)
    
    X_test_scaled = None
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler


def create_internal_validation_split(X_url, X_num, y, val_fraction=0.15, random_state=42):
    """
    Create internal validation split from training data.
    
    Args:
        X_url: URL strings
        X_num: Numerical features
        y: Labels
        val_fraction: Fraction for validation (default: 0.15)
        random_state: Random seed
    
    Returns:
        Training and validation splits for URLs, numerical features, and labels
    """
    X_url_train, X_url_val, X_num_train, X_num_val, y_train, y_val = train_test_split(
        X_url, X_num, y,
        test_size=val_fraction,
        stratify=y,
        random_state=random_state
    )
    
    print(f"✓ Internal validation split created:")
    print(f"  Training: {len(y_train)} samples")
    print(f"  Validation: {len(y_val)} samples")
    
    return X_url_train, X_url_val, X_num_train, X_num_val, y_train, y_val
