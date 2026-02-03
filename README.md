# URL Phishing Detection - Ensemble Deep Learning Framework

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19.0-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A robust, production-ready ensemble deep learning framework for detecting phishing URLs using advanced neural network architectures with comprehensive statistical validation.

✨ Key Features

Ensemble Learning

CNN–BiLSTM base model

Multi-scale CNN architecture

Attention-based neural model

Wide (high-capacity) architecture

Feature Engineering

Character-level URL tokenization

Vectorized numerical URL features

Fold-isolated feature extraction (no data leakage)

Evaluation & Validation

Stratified K-fold cross-validation (default: 10-fold)

Accuracy, Precision, Recall, F1-score, AUC-ROC

Statistical significance testing (McNemar, t-test, Wilcoxon, ANOVA, Cohen’s d)

Research-Oriented Design

Fully modular codebase

Deterministic training via fixed random seeds

Clear separation of data processing, modeling, and evaluation



## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```
git clone https://github.com/DenizKaya18/phishing-url-detection.git
cd url-phishing-detection
```

2. Install dependencies:
```
pip install -r requirements.txt
```

### Requirements

```
tensorflow==2.19.0
pandas==2.2.2
numpy==2.0.2
scikit-learn==1.6.1
wordsegment==1.3.1
tldextract==5.3.1
matplotlib==3.10.0
seaborn==0.13.2
psutil==5.9.5
```

## ⚡ Quick Start

### Basic Usage

```
from preprocessing import prepare_data_from_raw
from ensemble_classifier import OptimizedEnsembleURLClassifierCV

# Load dataset
X_train, X_test, y_train, y_test, tokenizer, max_len, vocab = \
    prepare_data_from_raw("data/dataset.txt")

# Initialize ensemble
classifier = OptimizedEnsembleURLClassifierCV(
    n_models=4,
    n_folds=10,
    random_seeds=[42, 123, 456, 789]
)

classifier.tokenizer = tokenizer
classifier.max_len = max_len
classifier.vocab_size = vocab

# Cross-validation
classifier.cross_validate_ensemble(X_train, y_train, epochs=15)

# Final training
classifier.train_final_ensemble(
    X_train, y_train, X_test, y_test, epochs=15
)

```

## 📁 Project Structure

```
url-phishing-detection/
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── ensemble_classifier.py
│   ├── evaluation.py
│   ├── statistical_tests.py
│	├── feature_extraction.py
│   └── main.py
│
├── data/
│   ├── README.md        👈 dataset description and source
│   └── raw/
│       └── mendeley_urls.txt
│
├── results/  # Evaluation outputs (auto-generated)
├── models/   # Saved models (auto-generated)
├── requirements.txt
├── README.md
└── LICENSE

```



### Dataset

This study uses a publicly available phishing URL dataset from Mendeley Data:

Source: https://data.mendeley.com/datasets/vfszbj9b36/1

Original labels: legitimate, phishing

Encoded labels:

0 → legitimate

1 → phishing

Label encoding was performed without altering class semantics or sample distribution.

Further details are provided in data/README.md.



### Training Pipeline

The `main.py` script executes a complete training pipeline:

1. **Data Preprocessing**: Load, tokenize, and split data
2. **Cross-Validation**: Train and evaluate models using K-fold CV
3. **Final Training**: Train ensemble on full training set
4. **Statistical Testing**: Validate model performance significance
5. **Results Summary**: Display comprehensive metrics

### Custom Configuration

```python
classifier, stats = main(
    data_file="data/dataset.txt",
    test_size=0.2,          # Test set proportion
    n_folds=10,             # Number of CV folds
    epochs=15,              # Training epochs
    batch_size=512          # Batch size
)
```

## 🏗️ Model Architectures

### 1. Base Model (CNN-BiLSTM)
- **Embedding**: 64-dimensional character embeddings
- **CNN**: 64 filters, kernel size 3, L2 regularization
- **BiLSTM**: 32 units per direction, dropout 0.3
- **Dense**: Two layers (64→32 units) with batch normalization

### 2. Multi-CNN Model
- Multiple convolutional branches (kernel sizes: 3, 5)
- Parallel feature extraction at different scales
- BiLSTM for sequence modeling
- Feature concatenation and fusion

### 3. Attention Model
- Custom attention mechanism for character importance
- CNN for local pattern detection
- Attention-weighted feature aggregation
- Dense classification layers

### 4. Wide Model
- Increased capacity (64 CNN filters, 64 BiLSTM units)
- Enhanced feature representation
- Suitable for complex pattern recognition
- Balanced regularization

## 🔍 Feature Extraction

- Character-level URL sequences (deep models)

- Vectorized numerical features (URL structure-based)

- Fold-isolated feature computation to prevent information leakage

- Standardized scaling applied only on training folds

## 📊 Evaluation Metrics

The framework computes comprehensive metrics:

- **Accuracy**: Overall classification accuracy
- **Precision**: Positive predictive value
- **Recall (Sensitivity)**: True positive rate
- **F1-Score**: Harmonic mean of precision and recall
- **Specificity**: True negative rate
- **FPR**: False positive rate
- **FNR**: False negative rate
- **AUC-ROC**: Area under the ROC curve
- **Confusion Matrix**: Detailed classification breakdown



## 📈 Statistical Tests

### Implemented Tests

1. **McNemar's Test**
   - Compares paired predictions
   - Tests for significant differences between models

2. **Paired t-Test**
   - Compares mean performance across folds
   - Parametric test for normally distributed data

3. **Wilcoxon Signed-Rank Test**
   - Non-parametric alternative to t-test
   - Robust to non-normal distributions



### Running Statistical Tests

```python
from statistical_tests import run_statistical_tests

# After training ensemble
statistical_results = run_statistical_tests(classifier)
```

## ⚙️ Configuration

### Model Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_models` | 4 | Number of models in ensemble |
| `n_folds` | 10 | Cross-validation folds |
| `epochs` | 15 | Training epochs per model |
| `batch_size` | 512 | Training batch size |
| `learning_rate` | 0.008 | Adam optimizer learning rate |
| `max_len` | Auto (95%ile) | Maximum sequence length |
| `embedding_dim` | 64 | Character embedding dimension |

### Training Options

```python
# Enable mixed precision training (automatic)
# Supports GPU acceleration
# Dynamic loss scaling for numerical stability

# Callbacks (built-in):
# - EarlyStopping (patience=5)
# - ReduceLROnPlateau (patience=3)
```

### Output Example

```
════════════════════════════════════════════════════════════════════════════════
                    URL PHISHING DETECTION - ENSEMBLE DEEP LEARNING                    
════════════════════════════════════════════════════════════════════════════════

📂 STEP 1: Data Preprocessing
────────────────────────────────────────────────────────────────────────────────
✓ Data loaded successfully
  Vocabulary size: 98
  Max sequence length: 167
  Training samples: 8000
  Test samples: 2000

🔬 STEP 2: Initialize Ensemble Classifier
────────────────────────────────────────────────────────────────────────────────

📊 STEP 3: Cross-Validation Training
────────────────────────────────────────────────────────────────────────────────

📈 Cross-Validation Results:
────────────────────────────────────────────────────────────────────────────────
  base        : 0.9650 (± 0.0089)
  multi_cnn   : 0.9625 (± 0.0095)
  attention   : 0.9638 (± 0.0092)
  wide        : 0.9668 (± 0.0085)
  Ensemble    : 0.9725 (± 0.0078)

✅ TRAINING COMPLETED SUCCESSFULLY
════════════════════════════════════════════════════════════════════════════════
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Deniz Kaya**




