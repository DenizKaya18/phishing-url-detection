# Dataset Description

This project uses a publicly available phishing URL dataset obtained from Mendeley Data:

**Original Dataset**
- Title: Phishing URL dataset
- Source: Mendeley Data
- URL: https://data.mendeley.com/datasets/vfszbj9b36/1

---

## Original Labels

In the original dataset, class labels are provided as categorical strings:

- `legitimate`
- `phishing`

---

## Label Transformation

For compatibility with binary classification models and deep learning frameworks, the original labels were transformed as follows:

| Original Label | Transformed Label |
|---------------|------------------|
| legitimate    | 0                |
| phishing      | 1                |

This transformation is **purely a numerical encoding step** and does **not** alter the semantic meaning or class distribution of the dataset.

---

## Dataset Format Used in This Project

After preprocessing, the dataset is stored in a plain text format where each line contains: <URL>,<label>

Example:
http://example.com,0
http://secure-login-update.xyz,1


- `0` → Legitimate (benign) URL  
- `1` → Phishing (malicious) URL  

---

## Preprocessing Notes

- No samples were removed during label transformation.
- No relabeling or class balancing was performed at this stage.
- The transformation ensures compatibility with:
  - TensorFlow / Keras
  - scikit-learn classifiers
  - Statistical significance tests

---

## Citation

If you use this dataset, please cite the original source:

Mendeley Data, Phishing Websites Dataset,
https://data.mendeley.com/datasets/vfszbj9b36/1

---

## Disclaimer

This dataset is used strictly for research and educational purposes.