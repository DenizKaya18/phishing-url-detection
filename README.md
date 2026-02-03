<<<<<<< HEAD
# Ensemble Deep Learning for URL Phishing Detection

A modular deep learning framework for detecting phishing URLs using ensemble of CNN-BiLSTM architectures with comprehensive feature engineering and statistical validation.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architectures](#model-architectures)
- [Evaluation Metrics](#evaluation-metrics)
- [Statistical Testing](#statistical-testing)
- [Requirements](#requirements)

## 🎯 Overview

This project implements an ensemble of four deep learning architectures for URL phishing detection:

1. **Base Model**: CNN + Bidirectional LSTM with L2 regularization
2. **Multi-CNN Model**: Multi-scale CNN with different kernel sizes
3. **Attention Model**: Self-attention mechanism with CNN
4. **Wide Model**: Increased capacity CNN-BiLSTM

The models are trained using 10-fold cross-validation and combined via soft voting for final predictions.

## ✨ Features

- **Modular Architecture**: Clean separation of preprocessing, modeling, and evaluation
- **Ensemble Learning**: Combines 4 different architectures for robust predictions
- **Cross-Validation**: 10-fold stratified CV for reliable performance estimates
- **Feature Engineering**: 20 handcrafted features including:
  - URL length and special character ratios
  - TLD-based features with learned weights
  - Bag-of-Words with segmentation
  - Character n-grams (3-grams and 4-grams)
  - Domain and path ratios
- **Statistical Validation**: Comprehensive statistical tests including:
  - McNemar's test for classifier comparison
  - Paired t-test and Wilcoxon test
  - ANOVA for multiple model comparison
  - Cohen's d effect size calculations
- **GPU Optimization**: Mixed precision training for faster computation

## 📁 Project Structure

```
.
├── src/
│   ├── preprocessing.py      # Data loading and preprocessing
│   ├── model.py              # Model architectures
│   ├── evaluation.py         # Metrics and visualization
│   ├── statistical_tests.py  # Statistical significance tests
│   └── main.py               # Main execution script
├── data/
│   └── Yeniyirmibin_dataset_V1.txt  # Dataset (not included)
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## 🚀 Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional, but recommended)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd url-phishing-detection
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### GPU Setup (Optional)

For GPU acceleration with mixed precision:

```bash
pip install tensorflow-gpu>=2.10.0
```

Ensure you have:
- CUDA Toolkit 11.2+
- cuDNN 8.1+

## 💻 Usage

### Basic Usage

```python
from preprocessing import prepare_data_from_raw
from evaluation import evaluate_model

# 1. Load and preprocess data
X_train, X_test, y_train, y_test, tokenizer, max_len, vocab_size = \
    prepare_data_from_raw("data/Yeniyirmibin_dataset_V1.txt")

# 2. Train model (requires full implementation)
# See original code for complete training pipeline

# 3. Evaluate
y_pred = model.predict(X_test)
results = evaluate_model(y_test, y_pred, model_name="Ensemble")
```

### Running the Complete Pipeline

```bash
python src/main.py
```

### Feature Extraction Example

The project uses isolated feature extraction for each fold:

```python
from feature_extraction import VectorizedFeatureExtractor

extractor = VectorizedFeatureExtractor(
    bow_data=bow_weights,
    seg_bow_data=segmented_bow_weights,
    ngrams_data=ngrams_weights,
    grams4_data=grams4_weights,
    tld_data=tld_weights
)

X_features, y_labels, processed_urls = extractor.extract_batch_vectorized(
    urls, labels, batch_size=5000
)
```

## 🏗️ Model Architectures

### 1. Base Model (CNN + BiLSTM)

```
Input (URL chars) → Embedding(64) → [CNN(64,3) + BiLSTM(32)] → Concat
Input (Features) → Dense(32) → BN → Dropout(0.3) → Concat
Merged → Dense(64) → BN → Dropout(0.4) → Dense(32) → BN → Dropout(0.3) → Output
```

### 2. Multi-CNN Model

Multiple parallel CNN branches with different kernel sizes (3 and 5) for multi-scale feature extraction.

### 3. Attention Model

Self-attention mechanism to focus on important URL segments.

### 4. Wide Model

Increased network capacity with wider layers (64 filters, 64 LSTM units).

## 📊 Evaluation Metrics

The framework calculates comprehensive metrics:

- **Classification Metrics**:
  - Accuracy
  - Precision
  - Recall (Sensitivity)
  - F1-Score
  - Specificity
  - False Positive Rate (FPR)
  - False Negative Rate (FNR)

- **Confusion Matrix**: Detailed breakdown of predictions

- **ROC Curve**: With AUC score

- **Cross-Validation Statistics**: Mean, std, min, max across folds

## 📈 Statistical Testing

### Implemented Tests

1. **McNemar's Test**
   - Compares two classifiers on the same test set
   - Tests if disagreements are significantly different

2. **Paired t-test**
   - Compares mean performance across CV folds
   - Assumes normal distribution

3. **Wilcoxon Signed-Rank Test**
   - Non-parametric alternative to t-test
   - More robust to outliers

4. **One-way ANOVA**
   - Compares multiple models simultaneously
   - Tests if at least one model differs significantly

5. **Cohen's d Effect Size**
   - Quantifies magnitude of difference
   - Interpretations: negligible (<0.2), small (0.2-0.5), medium (0.5-0.8), large (>0.8)

### Running Statistical Tests

```python
from statistical_tests import run_statistical_tests

# After training with cross-validation
statistical_results = run_statistical_tests(classifier)
```

## 📦 Requirements

### Core Dependencies

- `tensorflow>=2.10.0` - Deep learning framework
- `numpy>=1.21.0` - Numerical computing
- `pandas>=1.3.0` - Data manipulation
- `scikit-learn>=1.0.0` - ML utilities and metrics
- `scipy>=1.7.0` - Statistical functions

### Visualization

- `matplotlib>=3.4.0` - Plotting
- `seaborn>=0.11.0` - Statistical visualizations

### Feature Engineering

- `tldextract>=3.1.0` - TLD extraction
- `wordsegment>=0.2.0` - Word segmentation

See `requirements.txt` for complete list.

## 🎓 Citation

If you use this code in your research, please cite:

```bibtex
@article{url_phishing_detection_2024,
  title={Ensemble Deep Learning for URL Phishing Detection with Statistical Validation},
  author={Your Name},
  year={2024}
}
```

## 📝 License

[Add your license here]

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📧 Contact

[Add contact information]

## 🙏 Acknowledgments

- Dataset: Yeniyirmibin_dataset_V1
- Inspired by research in phishing detection and ensemble learning

---

**Note**: This is a modularized version of the original Colab notebook. The complete feature extraction pipeline and training loop require additional components from the original implementation.
=======
# phishing-url-detection
Source code for the paper: A Feature-Enriched Deep Learning Based Ensemble Framework for Robust Phishing URL Detection
>>>>>>> dc226492a83f3b58895dacc2a89b510a356a1579
