<div align="center">

# Heart Disease Prediction System

[![Python](https://img.shields.io/badge/Python-3.10+-3670A0?style=flat-square&logo=python&logoColor=ffdd54)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org)
[![Status](https://img.shields.io/badge/Status-Completed-2ea44f?style=flat-square)]()

**End-to-end ML pipeline for heart disease risk prediction using patient medical data.**

[Overview](#-overview) · [Pipeline](#-ml-pipeline) · [Dataset](#-dataset) · [Setup](#-installation) · [Usage](#-usage) · [Results](#-results)

</div>

---

## Overview

Heart disease is among the leading causes of death worldwide. Early and accurate prediction can significantly improve treatment outcomes.

This project implements a complete supervised learning pipeline — from raw medical data through EDA, feature engineering, multi-model training, evaluation, and a real-time Streamlit web interface for inference.

| Capability | Details |
|---|---|
| EDA & Visualization | Correlation heatmaps, distribution plots, statistical summary |
| Preprocessing | Missing value handling, StandardScaler feature normalization |
| Models Trained | Logistic Regression, Random Forest, SVM |
| Evaluation | Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix |
| Persistence | Joblib serialization (`.pkl`) for model + scaler |
| Deployment | Interactive Streamlit web app with real-time inference |

---

## ML Pipeline

```
Medical Dataset (UCI Heart Disease)
         │
         ▼
  Data Cleaning & EDA          ← Missing values, distributions, correlations
         │
         ▼
  Feature Engineering          ← Encoding, StandardScaler normalization
         │
         ▼
  Train / Test Split
         │
         ▼
  Model Training               ← Logistic Regression · Random Forest · SVM
         │
         ▼
  Evaluation & Selection       ← Accuracy, F1, ROC-AUC, Confusion Matrix
         │
         ▼
  Model Serialization          ← heart_model.pkl + scaler.pkl
         │
         ▼
  Streamlit Deployment         ← Real-time prediction interface
```

---

## Dataset

**Source:** [UCI Machine Learning Repository — Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)

### Input Features

| Feature | Description |
|---|---|
| `age` | Patient age |
| `sex` | Gender (0 = Female, 1 = Male) |
| `cp` | Chest pain type (0–3) |
| `trestbps` | Resting blood pressure (mm Hg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl |
| `restecg` | Resting ECG results |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina |
| `oldpeak` | ST depression induced by exercise |
| `slope` | Slope of peak exercise ST segment |
| `ca` | Number of major vessels (0–3) |
| `thal` | Thalassemia type |

**Target:** `0` = No Disease · `1` = Disease Present

---

## Project Structure

```
CodeAlpha_DiseasePrediction/
│
├── data/
│   └── heart.csv
│
├── models/
│   ├── heart_model.pkl        # Trained classifier
│   └── scaler.pkl             # Fitted StandardScaler
│
├── app/
│   └── app.py                 # Streamlit frontend
│
├── notebooks/
│   └── analysis.ipynb         # EDA and experimentation
│
├── train.py                   # Training pipeline
├── predict.py                 # CLI inference script
├── requirements.txt
└── README.md
```

---

## Installation

### 1 · Clone the Repository

```bash
cd CodeAlpha_DiseasePredictionfromMedicalData
```

### 2 · Create & Activate Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3 · Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### Train Models

```bash
python train.py
```

Runs the full pipeline: preprocessing → feature scaling → training → evaluation → saves `heart_model.pkl` and `scaler.pkl` to `models/`.

### CLI Prediction

```bash
python predict.py
```

### Streamlit Web App

```bash
streamlit run app/app.py
# → http://localhost:8501
```

---

## Results

| Model | Accuracy | Notes |
|---|---|---|
| Logistic Regression | 84–87% | Baseline linear classifier |
| Random Forest | 88–92% | Best performer; also provides feature importance |
| SVM | 85–90% | Strong non-linear boundary detection |

**Evaluation metrics used:** Accuracy · Precision · Recall · F1-Score · ROC-AUC · Confusion Matrix

**Best model selected** automatically based on ROC-AUC and serialized for deployment.

---

## Key Technical Decisions

**Why StandardScaler?**
SVM and Logistic Regression are sensitive to feature magnitude. Standardizing ensures stable gradient convergence and fair distance-based comparisons.

**Why Random Forest as the primary model?**
Handles non-linear feature interactions, is robust to outliers, and provides feature importance scores — critical for interpretability in a medical context.

**Why ROC-AUC over plain accuracy?**
Medical datasets are often class-imbalanced. ROC-AUC measures discrimination ability across all classification thresholds, making it a more reliable metric than raw accuracy.

---

## Future Improvements

- [ ] Hyperparameter tuning (GridSearchCV / Optuna)
- [ ] XGBoost and LightGBM integration
- [ ] SHAP explainability for clinical transparency
- [ ] FastAPI backend for REST inference endpoint
- [ ] Docker containerization
- [ ] Cloud deployment (AWS / Azure)
- [ ] CI/CD pipeline (GitHub Actions)

---

## Submission Checklist

- [x] Complete source code
- [x] Dataset included
- [x] Trained model artifacts
- [x] Streamlit deployment
- [x] README documentation
- [x] GitHub repository
- [x] LinkedIn explanation video

---

## Author

**Mohit Singh Rajput** — AI / ML Engineer

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat-square&logo=linkedin&logoColor=white)](https://linkedin.com/in/mohitsingh1307)
[![GitHub](https://img.shields.io/badge/GitHub-121011?style=flat-square&logo=github&logoColor=white)](https://github.com/Mohit-1307)
[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=flat-square&logo=kaggle&logoColor=white)](https://www.kaggle.com/mohitsinghrajput1307)
[![Email](https://img.shields.io/badge/Email-D14836?style=flat-square&logo=gmail&logoColor=white)](mailto:mohitsinghdausa@gmail.com)

---

<div align="center">

*If this project was useful, a ⭐ on the repository is appreciated.*

</div>
