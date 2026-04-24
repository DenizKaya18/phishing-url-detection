## 🧠 Machine Learning vs Deep Learning Baselines

This repository contains **two complementary pipelines** evaluated on the **same phishing URL dataset**:

### 🔹 Deep Learning Ensemble
- CNN–BiLSTM based architectures
- Attention mechanisms
- Multi-model ensemble with cross-validation
- Located under: `src/`

### 🔹 Classical Machine Learning Baselines
To provide a fair and reproducible comparison, we also include a **comprehensive classical ML pipeline** featuring:

- KNN, Random Forest, Gradient Boosting, Naive Bayes, MLP
- 10-fold stratified cross-validation
- Checkpointing and resume support
- Statistical significance testing

📁 **Location:** [`src_classical_ml/`](src_classical_ml/)  
📄 **Documentation:** [`src_classical_ml/README.md`](src_classical_ml/README.md)

Both pipelines:
- Use the **same dataset**
- Report identical evaluation metrics
- Enable direct and fair performance comparison


## 📁 Project Structure

```
url-phishing-detection/
├── src/                               # Deep Learning pipeline
│   ├── config.py                 	   # Global constants, hardware (GPU) settings, and paths
│   ├── utils.py                  	   # Logger class and time formatting helpers
│   ├── features.py               	   # VectorizedFeatureExtractor and IsolatedFeatureManager
│   ├── models.py                 	   # Hybrid Deep Learning architectures (CNN, LSTM, Attention)
│   ├── classifier.py             	   # OptimizedEnsembleURLClassifierCV (Main class logic)
│   ├── statistical_tests.py      	   # StatisticalSignificanceAnalyzer (McNemar, t-test, Wilcoxon)
│   └── main.py                   	   # Entry point for DL experiments and coordinate pipeline
│
├── src_classical_ml/                  # Classical Machine Learning pipeline
│   ├── __init__.py                    # Package initialization
│   ├── config.py                      # Global configuration and parameters
│   ├── data_loader.py                 # Dataset loading utilities
│   ├── preprocessor.py                # URL preprocessing and normalization
│   ├── feature_builder.py             # Handcrafted feature extraction
│   ├── models.py                      # ML model definitions (RF, KNN, MLP, etc.)
│   ├── trainer.py                     # Training and cross-validation pipeline
│   ├── evaluator.py                   # Model evaluation and metric computation
│   ├── checkpoint.py                  # Checkpointing and resume mechanism
│   ├── report_generator.py            # Result summarization and report creation
│   ├── requirements_classical_ml.txt  # Dependencies for classical ML pipeline
│   ├── run.py                         # Python execution script
│   ├── run.bat                        # Windows execution script
│   ├── run.sh                         # Linux execution script
│   ├── results/                       # Experimental outputs
│	├── README.md                      # Project overview and documentation
│   └── main.py                        # Entry point for classical ML experiments
│
├── data/
│   ├── README.md                      # Dataset description and source
│   └── dataset.txt                    # URL dataset (URL, label)
│
├── cv_checkpoints/                    # Auto-saved cross-validation states
├── models/                            # Trained models (auto-generated)
├── results/                      	   # Generated logs and statistical CSV reports
├── requirements.txt                   # Deep Learning project dependencies
├── README.md                          # Project overview and documentation
└── LICENSE

```



# URL Phishing Detection - Ensemble Deep Learning Framework

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19.0-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A robust, production-ready ensemble deep learning framework for detecting phishing URLs using advanced neural network architectures with comprehensive statistical validation.

✨ Key Features

- Ensemble Learning

- CNN–BiLSTM base model

- Multi-scale CNN architecture

- Attention-based neural model

- Wide (high-capacity) architecture

- Feature Engineering

- Character-level URL tokenization

- Vectorized numerical URL features

- Fold-isolated feature extraction (no data leakage)

- Evaluation & Validation

- Stratified K-fold cross-validation (default: 10-fold)

- Holdout Validation (80/20 train/test split) with dedicated Ablation Studies

- Accuracy, Precision, Recall, F1-score, AUC-ROC

- Statistical significance testing (McNemar, t-test, Wilcoxon, Friedman, Nemenyi)

- Research-Oriented Design

- Fully modular codebase

- Deterministic training via fixed random seeds

- Clear separation of data processing, modeling, and evaluation



## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/DenizKaya18/phishing-url-detection.git
	cd url-phishing-detection
    ```

2.  **Create and activate a virtual environment (Optional but recommended):**
    ```bash
    # Windows
    python -m venv venv
    .\venv\Scripts\activate

    # Linux/Mac
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### ☁️ Colab Setup

If you prefer to run the experiments in Google Colab, we have provided standalone scripts in the `colab_scripts/` directory:

1. Upload the `colab_scripts/` folder to your Google Drive.
2. Open `Classical_ML.py` or `DeepLearning.py` in Colab.
3. Run the respective requirements script (`ClasicalRequirements.py` or `DeepRequirements.py`) in a Colab cell to install necessary dependencies.
4. Mount your Google Drive and adjust the data paths to point to the uploaded dataset.


## ⚡ Quick Start

Running the Full Pipeline

To run the complete pipeline (Preprocessing → Cross-Validation → Final Training → Evaluation) with GPU optimization:

```bash
python -m src.main
```


### Programmatic Usage

You can import modules to run specific parts of the pipeline manually:

```
import os
from src.config import DATA_PATH, CHECKPOINT_DIR, EPOCHS, BATCH_SIZE
from src.classifier import OptimizedEnsembleURLClassifierCV
from src.models import CheckpointManager, extend_classifier_functionality

# 0. Inject saving utility to the class (Runtime Extension)
extend_classifier_functionality()

# 1. Initialize Classifier & Checkpoint Manager
classifier = OptimizedEnsembleURLClassifierCV(
    n_models=4,
    n_folds=10
)
checkpoint_mgr = CheckpointManager(checkpoint_dir=CHECKPOINT_DIR)

# 2. Load and Prepare Data
# Automatically fits tokenizer and determines max_len
X_url_train, y_train, X_url_test, y_test = classifier.prepare_data_from_raw(DATA_PATH)

# 3. Phase 1: Cross-Validation
# Uses local 'cv_checkpoints/' to resume if interrupted
classifier.cross_validate_ensemble(
    X_url_train, 
    y_train, 
    checkpoint_mgr=checkpoint_mgr,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

# 4. Phase 2: Final Ensemble Training
# Trains final models on the full training split
classifier.train_final_ensemble(
    X_url_train, y_train, 
    X_url_test, y_test,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

# 5. Phase 3: Evaluation & Detailed Reports
results, best_method = classifier.evaluate_final_ensemble(X_url_test, y_test)
classifier.print_comprehensive_summary(results, best_method)
classifier.print_cv_comparison(results, best_method)

# 6. Save Deployment Package
classifier.save_model_ensemble(save_path="models/ensemble_final")

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


## ⚙️ Configuration

### Model Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_models` | 4 | Number of models in ensemble |
| `n_folds` | 10 | Cross-validation folds |
| `epochs` | 15 | Training epochs per model |
| `batch_size` | 512 | Training batch size |
| `learning_rate` | 0.008 | Adam optimizer learning rate |
| `embedding_dim` | 64 | Character embedding dimension |

### 🚀 Training Optimizations

The framework automatically handles advanced training configurations:

* **Mixed Precision Training:** Uses `mixed_float16` policy for faster training and lower memory usage on supported GPUs.
* **Dynamic Loss Scaling:** Ensures numerical stability during half-precision training.
* **Built-in Callbacks:**
    * `EarlyStopping`: Monitors validation loss (patience=5) to prevent overfitting.
    * `ReduceLROnPlateau`: Reduces learning rate (factor=0.3, patience=2) when convergence stalls.


## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.






