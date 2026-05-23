# ==========================================
# HEART DISEASE PREDICTION PROJECT
# ==========================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

# ==========================================
# LOAD DATASET
# ==========================================

print("Loading dataset...")

# Load CSV
# Make sure heart.csv exists inside data folder

df = pd.read_csv("data/heart.csv")

print("\nFirst 5 Rows")
print(df.head())

# ==========================================
# BASIC INFORMATION
# ==========================================

print("\nDataset Shape")
print(df.shape)

print("\nDataset Info")
print(df.info())

print("\nStatistical Summary")
print(df.describe())

# ==========================================
# CHECK MISSING VALUES
# ==========================================

print("\nMissing Values")
print(df.isnull().sum())

# ==========================================
# DATA VISUALIZATION
# ==========================================

# Target Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='target', data=df)
plt.title('Heart Disease Distribution')
plt.xlabel('Target')
plt.ylabel('Count')
plt.show()

# Correlation Heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.show()

# ==========================================
# SPLIT FEATURES AND TARGET
# ==========================================

X = df.drop('target', axis=1)
y = df['target']

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# ==========================================
# FEATURE SCALING
# ==========================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, 'models/scaler.pkl')

print("\nScaler Saved Successfully")

# ==========================================
# MODEL TRAINING
# ==========================================

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42
    ),
    'SVM': SVC(probability=True)
}

best_model = None
best_accuracy = 0
best_model_name = ""

# ==========================================
# TRAIN & EVALUATE MODELS
# ==========================================

for name, model in models.items():

    print("\n" + "=" * 50)
    print(f"Training {name}")
    print("=" * 50)

    # Train model
    model.fit(X_train, y_train)

    # Prediction
    y_pred = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nAccuracy: {accuracy:.4f}")

    # Classification Report
    print("\nClassification Report")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, y_pred))

    # ROC AUC
    probabilities = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, probabilities)

    print(f"\nROC-AUC Score: {roc_auc:.4f}")

    # Save Best Model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_model_name = name
        

# ==========================================
# SAVE BEST MODEL
# ==========================================

joblib.dump(best_model, 'models/heart_model.pkl')

print("\n" + "=" * 50)
print("BEST MODEL")
print("=" * 50)
print(f"Model Name: {best_model_name}")
print(f"Accuracy: {best_accuracy:.4f}")

print("\nModel Saved Successfully")

# ==========================================
# FEATURE IMPORTANCE (Random Forest)
# ==========================================

if best_model_name == 'Random Forest':

    importances = best_model.feature_importances_

    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    })

    feature_importance = feature_importance.sort_values(
        by='Importance',
        ascending=False
    )

    print("\nFeature Importance")
    print(feature_importance)

    plt.figure(figsize=(10, 6))

    sns.barplot(
        x='Importance',
        y='Feature',
        data=feature_importance
    )

    plt.title('Feature Importance')
    plt.show()

print("\nTraining Completed Successfully")