# data processing
import pandas as pd
import numpy as np

# keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

# sklearn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, classification_report,
    precision_recall_curve, auc, confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.model_selection import train_test_split

# visualization
import matplotlib.pyplot as plt
import seaborn as sns

# misc
import os
import json

# function to load and preprocess quickdraw data
def load_quickdraw_data(dir, category_name, n_samples=1000):
    file = f'full_numpy_bitmap_{category_name}.npy'
    data = np.load(dir + file)
    if n_samples == -1:
        return data
    else:
        indices = np.random.choice(len(data), n_samples, replace=False)
        sampled_data = data[indices]
        return sampled_data
    

def preprocess_data(use_validation=True):
    # Data directory
    dir = '../data/'

    # Load and preprocess data
    all_data = []
    labels = []
    files = os.listdir(dir)
    categories = [file.split('_')[-1].split('.')[0] for file in files]  

    for category_name in categories:
        category_data = load_quickdraw_data(dir, category_name, 100) # Change this to -1 for all data
        all_data.extend(category_data)
        labels.extend([category_name] * len(category_data))

    if use_validation:
        train_data, val_data, train_labels, val_labels = train_test_split(all_data, labels, test_size=0.2, random_state=42)
        train_data, test_data, train_labels, test_labels = train_test_split(train_data, train_labels, test_size=0.25, random_state=42)

        # normalize data
        train_data = np.array(train_data).astype(np.float32) / 255
        val_data = np.array(val_data).astype(np.float32) / 255
        test_data = np.array(test_data).astype(np.float32) / 255

        # reshape
        train_data = train_data.reshape(-1, 784)
        val_data = val_data.reshape(-1, 784)
        test_data = test_data.reshape(-1, 784)

        # create true labels
        test_true_labels = np.ones(len(test_labels)).astype(bool)

        # convert to np array
        train_labels = np.array(train_labels)
        val_labels = np.array(val_labels)
        test_labels = np.array(test_labels)

        return train_data, test_data, val_data, train_labels, test_labels, val_labels, test_true_labels
    else:
        train_data, test_data, train_labels, test_labels = train_test_split(all_data, labels, test_size=0.25, random_state=42)

        # normalize data
        train_data = np.array(train_data).astype(np.float32) / 255
        test_data = np.array(test_data).astype(np.float32) / 255

        # reshape
        train_data = train_data.reshape(-1, 784)
        test_data = test_data.reshape(-1, 784)

        # create true labels
        test_true_labels = np.ones(len(test_labels)).astype(bool)

        # convert to np array
        train_labels = np.array(train_labels)
        test_labels = np.array(test_labels)

        return train_data, test_data, train_labels, test_labels, test_true_labels
    
def data_generator(data, batch_size=512):
    n_samples = len(data)
    for i in range(0, n_samples, batch_size):
        batch = data[i:i + batch_size]
        yield batch

def predict(model, data, threshold):
    reconstructions = model(data)
    loss = tf.keras.losses.mse(reconstructions, data)
    proba = np.mean(np.square(data - reconstructions), axis=1)
    return tf.math.less(loss, threshold), loss, proba


def get_metrics(predictions, labels):
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='binary')
    recall = recall_score(labels, predictions, average='binary')
    report = classification_report(labels, predictions, target_names=['anomaly', 'normal'])
    cm = confusion_matrix(labels, predictions)
    return accuracy, precision, recall, report, cm


def print_stats(accuracy, precision, recall, report, cm):
    print("Accuracy = {}".format(accuracy))
    print("Precision = {}".format(precision))
    print("Recall = {}".format(recall))
    print("Report\n {}".format(report))
    print("Confusion Matrix")
    print(cm)


def pr_auc_plot(proba, labels, is_machine_learning=False):
    if is_machine_learning:
        # Normalize the scores to a range [0, 1] for interpretability
        proba = 0.5 * (1 - (proba / np.max(np.abs(proba))))
        
    precision_curve, recall_curve, _ = precision_recall_curve(labels, proba)
    pr_auc = auc(recall_curve, precision_curve)

    # Plot the precision-recall curve
    plt.figure()
    plt.plot(recall_curve, precision_curve, color='darkorange', lw=2, label='PR Curve (area = %0.2f)' % pr_auc)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower right')
    plt.show()
    print()


def roc_plot(predictions, labels):
    unique_classes = np.unique(labels)

    if len(unique_classes) == 1:
        print("Only one class present in y_true. ROC AUC score is not defined in that case.")
    else:
        fpr, tpr, thresholds = roc_curve(labels, predictions)
        roc_auc = roc_auc_score(labels, predictions)

        # Plot ROC curve, FPR vs TPR, and True Positive Rate vs Threshold in a single plot
        plt.figure(figsize=(12, 8))

        # Plot ROC curve
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

        # Plot False Positive Rate vs Threshold
        plt.plot(thresholds, fpr, color='blue', lw=2, label='False Positive Rate (FPR)')

        # Plot True Positive Rate vs Threshold
        plt.plot(thresholds, tpr, color='green', lw=2, label='True Positive Rate (TPR)')

        # Set labels and title
        plt.xlabel('Threshold')
        plt.ylabel('Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve and Rates vs Threshold')
        plt.legend(loc='lower right')

        # Show the combined plot
        plt.show()


def boxplot_plot(title, scorer):
    # Plot a boxplot of the reconstruction errors
    plt.figure(figsize=(8, 6))
    sns.boxplot(y=scorer, color='skyblue')
    plt.title(f'{title} Distribution')
    plt.ylabel(f'{title}')
    plt.show()


def plot_anomaly_imgs(anomaly_indexes, data):
    # Plot some of the anomalies
    plt.figure(figsize=(12, 8))
    for i, idx in enumerate(anomaly_indexes[:10]):  # Plot the first 10 anomaly indices
        plt.subplot(2, 5, i + 1)
        plt.imshow(data[idx].reshape(28, 28), cmap='gray')
        plt.title(f'Anomaly {i+1}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def plot_all_anomaly_imgs(anomaly_indexes, data):
    # Plot all anomalies
    num_anomalies = len(anomaly_indexes)
    num_cols = 10  # You can adjust the number of columns based on your preference
    num_rows = (num_anomalies // num_cols) + (1 if num_anomalies % num_cols != 0 else 0)

    plt.figure(figsize=(15, 2 * num_rows))
    for i, idx in enumerate(anomaly_indexes):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(data[idx].reshape(28, 28), cmap='gray')
        plt.title(f'Anomaly {i + 1}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def create_heatmap(cm):
    tn, fp, fn, tp = cm.ravel()
    print(f"True Negatives: {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives: {tp}")
    sns.heatmap(cm, annot=True, fmt='d', cmap='magma', linewidths=1)


def write_to_json(preds, output_dir="."):
    # Get anomaly predictions
    anomaly_indexes = np.where(preds == False)[0]

    # Create a dictionary of indexes
    anomaly_dict = {"anomaly_indexes": list(map(int, anomaly_indexes))}

    # Construct the file path
    json_filename = os.path.join(output_dir, 'anomalies.json')

    # Save the anomaly dictionary to a JSON file
    with open(json_filename, 'w') as json_file:
        json.dump(anomaly_dict, json_file)

    print(f"Anomaly indexes saved to {json_filename}")
    return anomaly_indexes