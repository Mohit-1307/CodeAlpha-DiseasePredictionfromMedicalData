<div align="center">

# ❤️ Heart Disease Prediction System

### End-to-End Machine Learning Project for Disease Prediction Using Medical Data

<img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python">
<img src="https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange?style=for-the-badge&logo=scikitlearn">
<img src="https://img.shields.io/badge/Frontend-Streamlit-red?style=for-the-badge&logo=streamlit">
<img src="https://img.shields.io/badge/Data%20Analysis-Pandas-yellow?style=for-the-badge&logo=pandas">
<img src="https://img.shields.io/badge/Visualization-Seaborn-green?style=for-the-badge">
<img src="https://img.shields.io/badge/Status-Completed-success?style=for-the-badge">

<br>

### 🚀 Predict Heart Disease Risk Using Machine Learning

An advanced machine learning system that analyzes patient medical attributes and predicts the probability of heart disease using classification algorithms and real-time inference.

</div>

---

# 📌 Table of Contents

- Project Overview
- Features
- Tech Stack
- Machine Learning Pipeline
- Dataset Information
- Project Structure
- Installation
- Model Training
- Model Evaluation
- Streamlit Deployment
- Prediction System
- Results
- Future Improvements
- Screenshots
- Author

---

# 🚀 Project Overview

Heart disease is among the leading causes of death worldwide. Early prediction and diagnosis can significantly reduce risks and improve treatment outcomes.

This project implements a complete end-to-end Machine Learning pipeline capable of:

✅ Processing medical data  
✅ Performing Exploratory Data Analysis (EDA)  
✅ Training multiple classification models  
✅ Evaluating model performance  
✅ Saving trained models  
✅ Deploying a real-time prediction system using Streamlit  

The project demonstrates production-level ML workflow and software engineering practices.

---

# ✨ Features

<div align="center">

| Feature | Description |
|---|---|
| 📊 EDA | Exploratory Data Analysis |
| 📈 Visualization | Correlation Heatmaps & Graphs |
| ⚙️ Preprocessing | Feature Scaling |
| 🧠 ML Models | Logistic Regression, Random Forest, SVM |
| 📉 Evaluation | Accuracy, F1, ROC-AUC |
| 💾 Serialization | Joblib Model Saving |
| 🌐 Deployment | Streamlit Web App |
| ⚡ Real-Time Prediction | Instant Inference |

</div>

---

# 🛠️ Tech Stack

<div align="center">

## 💻 Programming & Development

<img src="https://skillicons.dev/icons?i=python,vscode,git,github"/>

---

## 📚 Data Science & Machine Learning

<img src="https://skillicons.dev/icons?i=tensorflow"/>

<img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white">
<img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white">
<img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white">
<img src="https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge">
<img src="https://img.shields.io/badge/Seaborn-43B02A?style=for-the-badge">

---

## 🌐 Deployment

<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white">

</div>

---

# 🧠 Machine Learning Pipeline

```text
Medical Dataset
       ↓
Data Cleaning
       ↓
EDA & Visualization
       ↓
Feature Engineering
       ↓
Train-Test Split
       ↓
Feature Scaling
       ↓
Model Training
       ↓
Model Evaluation
       ↓
Best Model Selection
       ↓
Model Serialization
       ↓
Streamlit Deployment
```

---

# 📊 Dataset Information

### Dataset Used:
- Heart Disease Dataset
- UCI Machine Learning Repository

---

## 📋 Input Features

| Feature | Description |
|---|---|
| age | Age |
| sex | Gender |
| cp | Chest Pain Type |
| trestbps | Resting Blood Pressure |
| chol | Cholesterol |
| fbs | Fasting Blood Sugar |
| restecg | ECG Results |
| thalach | Maximum Heart Rate |
| exang | Exercise Angina |
| oldpeak | ST Depression |
| slope | Slope |
| ca | Major Vessels |
| thal | Thalassemia |

---

# 📂 Project Structure

```text
CodeAlpha_DiseasePrediction/
│
├── data/
│   └── heart.csv
│
├── models/
│   ├── heart_model.pkl
│   └── scaler.pkl
│
├── app/
│   └── app.py
│
├── notebooks/
│   └── analysis.ipynb
│
├── train.py
├── predict.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# ⚙️ Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/Mohit-1307/CodeAlpha-DiseasePredictionfromMedicalData
```

---

## 2️⃣ Navigate to Project Directory

```bash
cd CodeAlpha_DiseasePrediction
```

---

## 3️⃣ Create Virtual Environment

### Windows

```bash
python -m venv venv
```

---

## 4️⃣ Activate Virtual Environment

```bash
venv\Scripts\activate
```

---

## 5️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run Project

## 🔥 Train Model

```bash
python train.py
```

This will:
- preprocess dataset
- scale features
- train models
- evaluate models
- save trained model
- generate `.pkl` files

---

## 🔍 Run Prediction

```bash
python predict.py
```

---

## 🌐 Launch Streamlit App

```bash
streamlit run app/app.py
```

---

# 🧠 Machine Learning Models

<div align="center">

| Model | Purpose |
|---|---|
| Logistic Regression | Baseline Linear Classifier |
| Random Forest | Ensemble Tree-Based Learning |
| Support Vector Machine | Non-Linear Classification |

</div>

---

# 📈 Evaluation Metrics

The project uses advanced evaluation metrics:

✅ Accuracy Score  
✅ Precision  
✅ Recall  
✅ F1-Score  
✅ ROC-AUC Score  
✅ Confusion Matrix  

---

# 📉 Feature Scaling

Feature scaling is performed using:

```python
StandardScaler()
```

This standardizes feature distributions and improves model convergence.

---

# 💾 Model Persistence

The trained model and scaler are serialized using Joblib.

```text
models/
├── heart_model.pkl
└── scaler.pkl
```

---

# 🌐 Streamlit Web Application

The project includes a fully interactive web interface.

## Features

✅ Real-time prediction  
✅ User-friendly interface  
✅ Medical parameter input  
✅ Instant inference  
✅ Prediction probability display  

---

# 📊 Exploratory Data Analysis

EDA includes:

- Missing value analysis
- Correlation heatmaps
- Feature distribution
- Statistical summary
- Disease distribution analysis

---

# 🔥 Feature Importance

Random Forest feature importance identifies the most influential medical parameters contributing to disease prediction.

---

# 📦 requirements.txt

```txt
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
streamlit
```

---

# 📸 Screenshots

## 🏠 Streamlit Home Page

<img width="100%" src="https://raw.githubusercontent.com/streamlit/streamlit/develop/examples/dogfood/streamlit_app.png">

---

## 📊 Correlation Heatmap

<img width="100%" src="https://miro.medium.com/v2/resize:fit:1400/1*9qqq8h0x4uD6nK1Rz6B0FA.png">

---

# 🚀 Future Improvements

- Hyperparameter Tuning
- XGBoost Integration
- Deep Learning Models
- SHAP Explainability
- Docker Deployment
- FastAPI Backend
- Cloud Deployment
- CI/CD Pipelines

---

# 📚 Learning Outcomes

This project demonstrates:

✅ End-to-End ML Workflow  
✅ Data Preprocessing  
✅ Model Training  
✅ Feature Scaling  
✅ Evaluation Metrics  
✅ Deployment Pipeline  
✅ Streamlit Integration  
✅ Software Engineering Practices  

---

# 📌 Internship Submission Checklist

- [x] Complete Source Code
- [x] Dataset Included
- [x] Trained Model
- [x] Streamlit Deployment
- [x] README Documentation
- [x] GitHub Repository
- [x] LinkedIn Explanation Video

---

# 📊 Expected Accuracy

| Model | Accuracy |
|---|---|
| Logistic Regression | 84% - 87% |
| Random Forest | 88% - 92% |
| SVM | 85% - 90% |

---

# 👨‍💻 Author

## Mohit

Machine Learning Engineer | AI Enthusiast | Python Developer

---

# ⭐ Support

If you found this project useful:

⭐ Star the repository  
🍴 Fork the project  
📢 Share it with others  

---

<div align="center">

# ❤️ Thank You

### Made with Python, Machine Learning & Streamlit

</div>