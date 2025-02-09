# Install necessary libraries if not already installed
try:
    import xgboost
except ModuleNotFoundError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost"])
    import xgboost

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
)
from sklearn.dummy import DummyClassifier
import joblib  # For saving and loading the model

# Step 1: Load the dataset
# Replace 'osteoporosis.csv' with your actual dataset file path
df = pd.read_csv('osteoporosis.csv')

# Step 2: Data Understanding & Exploration
print("First 5 rows of the dataset:")
print(df.head())

print("\nShape of the dataset (rows, columns):")
print(df.shape)

print("\nSummary statistics of the dataset:")
print(df.describe())

print("\nMissing values in each column:")
print(df.isnull().sum())

# Clean column names by stripping any leading/trailing spaces
df.columns = df.columns.str.strip()
print("\nCleaned column names:")
print(df.columns)

# Visualize the distribution of 'Age'
plt.figure(figsize=(10, 6))
sns.histplot(df['Age'], bins=20, kde=True)
plt.title('Distribution of Age')
plt.show()

# Visualize the distribution of 'Body Weight' (if the column exists)
if 'Body Weight' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Body Weight', data=df)
    plt.title('Distribution of Body Weight')
    plt.show()
else:
    print("\nColumn 'Body Weight' not found in DataFrame.")

# Visualize the distribution of 'Gender' (if the column exists)
if 'Gender' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Gender', data=df)
    plt.title('Distribution of Gender')
    plt.show()
else:
    print("\nColumn 'Gender' not found in DataFrame.")

# Visualize correlations between numerical features
numeric_df = df.select_dtypes(include=['number'])
plt.figure(figsize=(10, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Visualize pair plots with 'Osteoporosis' as hue (if the column exists)
if 'Osteoporosis' in df.columns:
    sns.pairplot(df, hue='Osteoporosis')
    plt.show()
else:
    print("\nColumn 'Osteoporosis' not found in DataFrame.")

# Step 3: Data Preprocessing
# Handle missing values
# For numeric columns, fill missing values with the median
numeric_columns = df.select_dtypes(include=['number']).columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())

# For categorical columns, fill missing values with the mode
categorical_columns = df.select_dtypes(include=['object']).columns
df[categorical_columns] = df[categorical_columns].fillna(df[categorical_columns].mode().iloc[0])

# Encode categorical variables using one-hot encoding
df = pd.get_dummies(df, drop_first=True)

# Step 4: Model Training
# Split the dataset into features (X) and target (y)
X = df.drop('Osteoporosis', axis=1)  # Features
y = df['Osteoporosis']  # Target variable

# Split the data into training and testing sets (80-20 split) with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize and train the XGBoost model
model = XGBClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 5: Model Evaluation
# Make predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probabilities for ROC-AUC

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

# Print evaluation metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not At Risk', 'At Risk'], yticklabels=['Not At Risk', 'At Risk'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Step 6: Comparison Against Baseline
# Baseline model (always predicts the majority class)
baseline_model = DummyClassifier(strategy='most_frequent')
baseline_model.fit(X_train, y_train)
y_pred_baseline = baseline_model.predict(X_test)

# Baseline accuracy
baseline_accuracy = accuracy_score(y_test, y_pred_baseline)
print(f"Baseline Accuracy: {baseline_accuracy:.4f}")

# Comparison
print("\nModel Comparison:")
print(f"XGBoost Accuracy: {accuracy:.4f}")
print(f"Baseline Accuracy: {baseline_accuracy:.4f}")

# Step 7: Save the Trained Model
model_filename = 'osteoporosis_risk_xgboost_model.pkl'
joblib.dump(model, model_filename)
print(f"Model saved to {model_filename}")

# Step 8: Load the Saved Model and Make Predictions
# Load the saved model
loaded_model = joblib.load(model_filename)
print("Model loaded successfully!")

# Make predictions using the loaded model
new_predictions = loaded_model.predict(X_test)
print("Predictions from the loaded model:")
print(new_predictions)