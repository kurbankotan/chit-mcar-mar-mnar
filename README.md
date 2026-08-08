# CHIT under MCAR, MAR, and MNAR

> **Comparison of the CHIT algorithm with other imputation algorithms under MCAR, MAR, and MNAR error patterns**

Source code for the paper:

**"Evaluating the Cyclical Hybrid Imputation Technique (CHIT) Under MCAR, MAR, and MNAR Missing Data Mechanisms: Evidence from Health Datasets"**

PLOS ONE — Manuscript ID: PONE-D-26-23591  
Corresponding author: Asst. Prof. Kurban Kotan — kurban.kotan@cbu.edu.tr  
Manisa Celal Bayar University, Department of Artificial Intelligence and Machine Learning

---

## Requirements

- **Google Colab Pro or Pro+** — High-RAM is required (≥40 GB recommended)
- **A100 GPU** recommended for DL experiments
- Python 3.10+, scikit-learn, TensorFlow 2.20, pandas, numpy, scipy

---

## Files

| File | Description |
|------|-------------|
| `CHIT_COMPLETE.py` | **Main script** — runs all 9 experiments (3 datasets × 3 mechanisms) |
| `CHIT_CKD.py` | CKD-only experiments (faster, ~15 min) |
| `CHIT_HDD.py` | HDD-only experiments (~45 min) |
| `CHIT_MPED_MAR_MNAR.py` | MPED MAR+MNAR only (~6-8 hours) |

---

## Datasets

All datasets are publicly available from the [UCI Machine Learning Repository](https://archive.ics.uci.edu):

| Dataset | n | Features | Reference |
|---------|---|----------|-----------|
| Chronic Kidney Disease (CKD) | 400 | 24 | [UCI CKD](https://archive.ics.uci.edu/dataset/336) |
| Heart Disease (HDD) | 1025 | 13 | [UCI Heart](https://archive.ics.uci.edu/dataset/45) |
| Mice Protein Expression (MPED) | 1080 | 82 | [UCI MPED](https://archive.ics.uci.edu/dataset/342) |

---

## Drive Paths (update if needed)

```python
CKD_PATH  = '/content/drive/MyDrive/Colab Notebooks/datasets/kidney_disease/kidney_disease.csv'
HDD_PATH  = '/content/drive/MyDrive/Colab Notebooks/datasets/Heart Disease Dataset/heart.csv'
MPED_PATH = '/content/drive/MyDrive/Colab Notebooks/datasets/Mice Protein Expression Dataset/Data_Cortex_Nuclear.csv'
DRIVE_OUT = '/content/drive/MyDrive/Colab Notebooks/'
```

---

## Pipeline

```
Raw data
  └─ Missing mask (MCAR / MAR / MNAR)
       ├─ CHIT  → StandardScaler → 42 configs (6 col-stage × 7 row-stage)
       ├─ MICE  → Raw data → BayesianRidge IterativeImputer
       ├─ RF-Iter → Raw data → RandomForest IterativeImputer
       └─ KNN   → Raw data → KNNImputer (k=5)
            └─ train_test_split (80:20, random_state=42)
                 └─ GridSearchCV (cv=5) → 8 classifiers
                      └─ Accuracy, F1, Precision, Recall, McNemar, Wilson CI
```

---

## Output

Each experiment produces a JSON file:

```
chit_complete_mcar.json        # CKD MCAR
chit_complete_mar.json         # CKD MAR
chit_complete_mnar.json        # CKD MNAR
chit_complete_hdd_mcar.json    # HDD MCAR
chit_complete_hdd_mar.json     # HDD MAR
chit_complete_hdd_mnar.json    # HDD MNAR
chit_complete_mped_mcar.json   # MPED MCAR
chit_complete_mped_mar.json    # MPED MAR
chit_complete_mped_mnar.json   # MPED MNAR
```

---

## Results Summary

| Dataset | Mechanism | CHIT Mean | MICE Mean | Gap |
|---------|-----------|-----------|-----------|-----|
| CKD | MCAR | 99.38% | 83.91% | +15.47 pp |
| CKD | MAR | 100.00% | 80.62% | +19.38 pp |
| CKD | MNAR | 99.06% | 85.94% | +13.12 pp |
| HDD | MCAR | 92.62% | 83.84% | +8.78 pp |
| HDD | MAR | 94.58% | 86.46% | +8.12 pp |
| HDD | MNAR | 94.76% | 88.54% | +6.22 pp |
| MPED | MCAR | 99.08% | 99.94% | −0.86 pp |
| MPED | MAR | 99.88% | 100.00% | −0.12 pp |
| MPED | MNAR | 99.88% | 100.00% | −0.12 pp |

> **Note:** MPED results show near-ceiling performance for all methods due to high-dimensional feature redundancy (82 protein features).

---

## Citation

If you use this code, please cite:

```
Kotan, K. (2026). Evaluating the Cyclical Hybrid Imputation Technique (CHIT) Under 
MCAR, MAR, and MNAR Missing Data Mechanisms: Evidence from Health Datasets. 
PLOS ONE. Manuscript ID: PONE-D-26-23591.
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
