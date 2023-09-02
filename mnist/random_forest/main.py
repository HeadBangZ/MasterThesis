"""
    Starting file to process mnist dataset 
    creates a decision tree using DecisionTreeClassifier
    with ensemble learning method random forest classifier.
"""
# Data processing
import pandas as pd
import numpy as np

# Modelling
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier

# Tree Visualization
from sklearn.tree import export_graphviz

x, y = load_digits(return_X_y=True)
rf = RandomForestClassifier(n_estimators=100)

X_train, X_test, Y_train, Y_test = train_test_split(
    x, y, test_size=.25, random_state=42)
X_train, X_val, Y_train, Y_val = train_test_split(
    x, y, test_size=.25, random_state=42)

print(f"Dataset split: {len(X_train)} train rows",
      f"{len(X_val)} valid rows",
      f"{len(Y_test)} test rows")
