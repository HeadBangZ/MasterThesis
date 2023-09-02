"""
    Starting file to process mnist dataset 
    creates a decision tree using RandomForestClassifier
    which is an ensemble learning method for decision trees.
"""
# Data processing
import pandas as pd
import numpy as np

# Modelling
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split

# Tree Visualization
import matplotlib.pyplot as plt
import seaborn as sn

# Classes
from data_processing import DataProcessing
from model_visualization import ModelVisualization


def init():
    """
        Initializes model
    """
    global dp
    global mv
    dp = DataProcessing(load_digits())
    mv = ModelVisualization()
    mnist = dp.get_data()

    pixels = dp.get_data_points()
    labels = dp.get_labels()

    mv.plot_image(dp.get_specific_data_point(
        pixels), dp.get_specific_label(labels))

    prepare_model(mnist)


def prepare_model(dataset):
    """
    Prepares the model
    Args:
        dataset: Dataset
    """
    X_train, X_test, Y_train, Y_test = train_test_split(
        dataset['data'], dataset['target'], test_size=.25, random_state=42)

    model = RandomForestClassifier(
        n_estimators=1000, max_depth=64, random_state=42, bootstrap=True, oob_score=True)

    print(f"Dataset split: {len(X_train)} train rows",
          f"{len(Y_test)} test rows")

    train_model(model, X_train, Y_train, X_test, Y_test)


def train_model(model, X_train, Y_train, X_test, Y_test):
    """
    Trains the model
    Args:
    Takes in X and Y train / test / val
    """
    model.fit(X_train, Y_train)
    score = model.score(X_train, Y_train)
    print(score)
    predict_model(model, X_test, Y_test)


def predict_model(model, X_test, Y_test):
    """
    Predicts the model result

    Args:
        model (_type_): dataset
        X_test (_type_): X_test data
        Y_test (_type_): Y_test data
    """
    prediction = model.predict(X_test)

    print("Accuracy")
    print(accuracy_score(prediction, Y_test))
    print("--------------------")
    print("Recall")
    print(recall_score(Y_test, prediction, average=None))
    print("--------------------")
    print("Precision")
    print(precision_score(Y_test, prediction, average=None))
    print("--------------------")

    print(X_test[0])
    print(Y_test[0])

    model.predict(X_test[0].reshape(1, -1))

    mv.plot_image(X_test[0], Y_test[0])
    result(Y_test, prediction)


def result(Y_test, prediction):
    """
    Makes result of the model

    Args:
        Y_test (_type_): Y_test data
        prediction (_type_): model prediction
    """
    cm = confusion_matrix(Y_test, prediction)
    print(cm)

    mv.plot_heatmap(cm)
