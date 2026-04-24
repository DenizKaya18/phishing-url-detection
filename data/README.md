# Dataset Description
This project uses three publicly available phishing URL datasets for training, evaluation, and cross-dataset generalization experiments.

## Datasets Overview

| ID | Name | Source | Year |
|---|---|---|---|
| D1 | Mendeley Phishing URL Dataset (V1) | Mendeley Data | 2024 |
| D2 | StealthPhisher Dataset | Mendeley Data | 2025 |
| D3 | 1M-PD (1 Million Phishing Dataset) | GitHub – ChracterEmbedding | 2018 |

### D1: Mendeley Phishing URL Dataset (V1)

**Title:** Phishing URL Dataset  
**Source:** Mendeley Data  
**URL:** https://data.mendeley.com/datasets/vfszbj9b36/1  

**Original Labels**

| Original Label | Transformed Label |
|---|---|
| legitimate | 0 |
| phishing | 1 |

### D2: StealthPhisher Dataset (2025)

**Title:** StealthPhisher  
**Source:** Mendeley Data  
**URL:** https://data.mendeley.com/datasets/m2479kmybx/1  

**Original Labels**

| Original Label | Transformed Label |
|---|---|
| legitimate | 0 |
| phishing | 1 |

### D3: 1M-PD Dataset (2018)

**Title:** 1M-PD (Character Embedding Phishing Dataset)  
**Source:** GitHub – huapingz/ChracterEmbedding  
**URL:** https://github.com/huapingz/ChracterEmbedding/tree/master  

**Original Labels**

| Original Label | Transformed Label |
|---|---|
| legitimate | 0 |
| phishing | 1 |

## Label Transformation
For compatibility with binary classification models and deep learning frameworks, all dataset labels were transformed to a unified binary encoding:

- **0** → Legitimate (benign) URL
- **1** → Phishing (malicious) URL

This transformation is purely a numerical encoding step and does not alter the semantic meaning or class distribution of any dataset.

## Dataset Format Used in This Project
After preprocessing, all datasets are stored in a plain text format where each line contains:
`<URL>,<label>`

**Example:**
```text
http://example.com,0
http://secure-login-update.xyz,1
```

## Preprocessing Notes

- No samples were removed during label transformation.
- No relabeling or class balancing was performed at this stage.
- All three datasets were preprocessed independently before being used in within-dataset and cross-dataset evaluation protocols.
- The unified format ensures compatibility with:
  - TensorFlow / Keras
  - scikit-learn classifiers
  - Statistical significance tests (McNemar, Wilcoxon)

## Citation
If you use any of these datasets, please cite the original sources:

**D1:**  
Mendeley Data, Phishing Websites Dataset, https://data.mendeley.com/datasets/vfszbj9b36/1

**D2:**  
Mendeley Data, StealthPhisher, https://data.mendeley.com/datasets/m2479kmybx/1

**D3:**  
Huang, P. et al. (2018). Character-level based Detection of Phishing Webpages. GitHub: https://github.com/huapingz/ChracterEmbedding

## Disclaimer
All datasets are used strictly for research and educational purposes.