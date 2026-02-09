# Classical ML for Phishing URL Detection

This module implements classical machine learning models for phishing URL detection with comprehensive cross-validation, checkpointing, and reporting capabilities.

## Features

- **Multiple ML Models**: KNN, Random Forest, Gradient Boosting, Naive Bayes, MLP
- **10-Fold Cross-Validation**: Stratified K-Fold for robust evaluation
- **Automatic Checkpointing**: Resume training from interruptions
- **Parallel Processing**: Efficient multi-core utilization
- **Comprehensive Reporting**: Performance metrics, confusion matrices, statistical tests
- **Local Output**: Results saved to project output directory

## Project Structure

```
src/classical_ml/
├── __init__.py              		# Package initialization
├── config.py                		# Configuration management
├── data_loader.py           		# Data loading utilities
├── preprocessor.py          		# URL preprocessing
├── feature_builder.py       		# Feature extraction
├── models.py                		# Model definitions
├── trainer.py               		# Training pipeline
├── evaluator.py             		# Model evaluation
├── checkpoint.py            		# Checkpoint management
├── report_generator.py      		# Report generation
├── requirements_classical_ml.txt   # Dependencies for classical ML pipeline
├── run.py                          # Python execution script
├── run.bat                         # Windows execution script
├── run.sh                          # Linux execution script
├── results/                        # Experimental outputs
├── README.md                       # Project overview and documentation
└── main.py		

```

## Requirements

```bash
pip install -r requirements_classical_ml.txt
```

Or install manually:
```bash
pip install numpy pandas scikit-learn imbalanced-learn seaborn matplotlib scipy tqdm joblib
pip install tldextract wordsegment  
```

## Usage

### Standard Python Environment

**Recommended Method:**
```bash
# Navigate to classical_ml directory
cd src/classical_ml

# Run the pipeline (all platforms)
python run.py
```

**Alternative Methods:**

Windows:
```cmd
cd src\classical_ml
run.bat
```

Linux/Mac:
```bash
cd src/classical_ml
./run.sh
```

### Configuration

Edit `config.py` to customize:

```python
class Config:
    def __init__(self, output_folder="../../output/classical_ml"):
        self.output_folder = output_folder       # Output directory
        self.n_splits = 10                       # Number of CV folds
        self.random_state = 42                   # Random seed
        self.feature_threshold = 20              # Feature frequency threshold
        self.preprocessing_batch_size = 5000     # Batch size for preprocessing
```

## Features Extracted

The pipeline extracts 20 features from each URL:

1. **Basic Features**:
   - URL length
   - Special character count
   - Special character ratio
   
2. **Domain Features**:
   - TLD weight
   - Dot count in netloc
   - IP address presence
   
3. **Security Indicators**:
   - URL shortener usage
   - @ symbol presence
   - Double slash occurrence
   - Dashboard in netloc
   - Non-standard port
   - https- pattern
   - Executable file extension
   
4. **Linguistic Features**:
   - Unique tokens count
   - Bag-of-words score
   - Segmented bag-of-words score
   - 3-gram score
   - 4-gram score
   
5. **Structural Features**:
   - Path to URL length ratio
   - Netloc to URL length ratio

## Output Files

All results are saved to the output directory (default: `../../output/classical_ml/`):

1. **Detailed_Performance_Report_Per_Fold.csv**: Per-fold metrics for all models
2. **Baseline_Average_Performance.csv**: Averaged performance across folds
3. **CM_Baseline_*.png**: Confusion matrix visualizations
4. **Statistical_Significance_Report.txt**: Statistical comparisons (t-test, Wilcoxon, ANOVA, Cohen's d)
5. **Checkpoints**: Automatic partial results for resuming

## Metrics Reported

- Accuracy, Precision, Recall, F1-Score
- Sensitivity (TPR), Specificity (TNR)
- False Negative Rate (FNR), False Positive Rate (FPR)
- Training time per fold
- Average single prediction time (μs precision)
- Confusion matrix components (TN, FP, FN, TP)

## Checkpointing

The pipeline automatically saves progress after each model training:

- **Partial CSV**: Contains all completed model evaluations
- **Metadata JSON**: Tracks completed folds
- **Resume Support**: Automatically skips completed work on restart

To restart from checkpoint, simply run `main.py` again.

## Class Imbalance Handling

Two strategies are automatically attempted:

1. **Sample Weighting**: Balanced class weights during training
2. **Oversampling**: RandomOverSampler as fallback

## Statistical Analysis
Automatic statistical significance testing includes:

- **Friedman test** (non-parametric alternative to repeated-measures ANOVA)  
  → Tests whether there is a statistically significant difference among the models across the cross-validation folds

- **Nemenyi post-hoc test** (when Friedman p < 0.05)  
  → Performs pairwise comparisons to identify which specific models differ significantly from each other

- **Mean accuracy ranking** and descriptive statistics per model


## Performance Optimization

- **Parallel Processing**: Joblib parallelization across CPU cores
- **Batch Processing**: Large dataset handling with batching
- **Caching**: Preprocessed data cached for faster re-runs
- **Atomic Operations**: Safe checkpoint saves

## Notes

- **Data Path**: Dataset expected at `../../data/dataset.txt` (relative to src/classical_ml)
- **Output Path**: Results saved to `../../output/classical_ml/` (can be configured)
- **Format**: `URL,label` where label is 0 (benign) or 1 (phishing)
- **Atomic Operations**: Safe checkpoint saves

