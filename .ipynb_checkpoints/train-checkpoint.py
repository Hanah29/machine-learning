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
    print