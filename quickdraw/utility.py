# data processing
import pandas as pd
import numpy as np

# keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist

# sklearn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, classification_report,
    precision_recall_curve, auc, confusion_matrix, roc_auc_score, roc_curve, f1_score
)
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# visualization
import matplotlib.pyplot as plt
import seaborn as sns

# misc
import os
import json
import time
import psutil
import threading


# Add these functions to load and preprocess MNIST data
def load_mnist_data(n_samples=12000):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    data = np.concatenate([x_train, x_test], axis=0)
    labels = np.concatenate([y_train, y_test], axis=0)

    # Select a subset of MNIST digits
    indices = np.random.choice(len(data), n_samples, replace=False)
    sampled_data = data[indices]
    sampled_labels = labels[indices]

    return sampled_data, sampled_labels


def add_mnist_anomalies(test_data, test_labels, n_anomalies):
    # Load MNIST data
    mnist_data, mnist_labels = load_mnist_data(n_samples=n_anomalies)
    mnist_labels = mnist_labels.astype(int)

    # Reshape MNIST data to match the shape of your quickdraw data
    mnist_data = mnist_data.reshape(-1, 784).astype(np.float32) / 255

    # Add MNIST data to the test data
    test_data = np.concatenate([test_data, mnist_data], axis=0)

    # Save the original labels
    original_labels = test_labels.copy()

    # Concatenate MNIST labels and test labels
    all_labels = np.concatenate([original_labels, mnist_labels])

    shuffled_data, shuffled_labels = shuffle(test_data, all_labels, random_state=42)

    test_true_labels = np.where((shuffled_labels.astype(str) >= '0') & (shuffled_labels.astype(str) <= '9'), False, True)

    return shuffled_data, shuffled_labels, test_true_labels, mnist_data


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


def preprocess_data(use_validation=True, n_anomalies=12000, n_samples=10000):
    # Data directory
    dir = '../data/'

    # Load and preprocess data
    all_data = []
    labels = []
    files = os.listdir(dir)
    categories = [file.split('_')[-1].split('.')[0] for file in files]  
    print(len(categories))
    for category_name in categories:
        category_data = load_quickdraw_data(dir, category_name, n_samples) # Change this to -1 for all data
        all_data.extend(category_data)
        labels.extend([category_name] * len(category_data))

    if use_validation:
        print("Splitting data")
        train_data, val_data, train_labels, val_labels = train_test_split(all_data, labels, test_size=0.2, random_state=42)
        train_data, test_data, train_labels, test_labels = train_test_split(train_data, train_labels, test_size=0.25, random_state=42)

        # normalize data
        print("normalize data")
        train_data = np.array(train_data).astype(np.float32) / 255
        val_data = np.array(val_data).astype(np.float32) / 255
        test_data = np.array(test_data).astype(np.float32) / 255

        # reshape
        print("Reshape data")
        train_data = train_data.reshape(-1, 784)
        val_data = val_data.reshape(-1, 784)
        test_data = test_data.reshape(-1, 784)

        # convert to np array
        print("Convert data")
        train_labels = np.array(train_labels)
        val_labels = np.array(val_labels)
        test_labels = np.array(test_labels)

        # Add MNIST anomalies to the test data and labels
        print("Add anomalies")
        test_data, test_labels, test_true_labels, anom_data = add_mnist_anomalies(test_data, test_labels, n_anomalies)

        return train_data, test_data, val_data, train_labels, test_labels, val_labels, test_true_labels, anom_data
    else:
        print("Splitting data")
        train_data, test_data, train_labels, test_labels = train_test_split(all_data, labels, test_size=0.25, random_state=42)

        # normalize data
        print("normalize data")
        train_data = np.array(train_data).astype(np.float32) / 255
        test_data = np.array(test_data).astype(np.float32) / 255

        # reshape
        print("Reshape data")
        train_data = train_data.reshape(-1, 784)
        test_data = test_data.reshape(-1, 784)

        # convert to np array
        print("Convert data")
        train_labels = np.array(train_labels)
        test_labels = np.array(test_labels)

        # Add MNIST anomalies to the test data and labels
        print("Add anomalies")
        test_data, test_labels, test_true_labels, anom_data = add_mnist_anomalies(test_data, test_labels, n_anomalies)

        return train_data, test_data, train_labels, test_labels, test_true_labels, anom_data

    
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


def predict_in_batches(model, data, threshold, batch_size=4096):
    # Get the total number of samples
    total_samples = len(data)

    # Initialize arrays to store results
    all_losses = []
    all_probas = []

    # Process the data in batches
    for i in range(0, total_samples, batch_size):
        # Extract a batch
        batch_data = data[i:i+batch_size]

        # Perform reconstruction on the batch using the model
        batch_reconstructions = model(batch_data)

        # Calculate loss and probability for the batch
        batch_loss = tf.keras.losses.mse(batch_reconstructions, batch_data)
        batch_proba = np.mean(np.square(batch_data - batch_reconstructions), axis=1)

        # Append the batch results to the lists
        all_losses.append(batch_loss)
        all_probas.append(batch_proba)

    # Concatenate the results from all batches
    losses = tf.concat(all_losses, axis=0)
    probas = np.concatenate(all_probas, axis=0)

    return tf.math.less(losses, threshold), losses, probas


def get_metrics(predictions, labels):
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='binary')
    recall = recall_score(labels, predictions, average='binary')
    f1 = f1_score(labels, predictions, average='binary')
    report = classification_report(labels, predictions, target_names=['anomaly', 'normal'])
    cm = confusion_matrix(labels, predictions)
    return accuracy, precision, recall, f1, report, cm


def print_stats(accuracy, precision, recall, f1, report, cm):
    print("Accuracy = {}".format(accuracy))
    print("Precision = {}".format(precision))
    print("Recall = {}".format(recall))
    print("F1 = {}".format(f1))
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


def plot_normalized_pixel_data(normal_data, anomaly_data):
    plt.grid()
    plt.plot(np.arange(784), normal_data[-1])
    plt.title("Normal data")
    plt.show()

    plt.grid()
    plt.plot(np.arange(784), anomaly_data[-1])
    plt.title("Abnormal data")
    plt.show()


def boxplot_plot(title, scorer):
    # Plot a boxplot of the data
    plt.figure(figsize=(8, 6))
    boxplot = plt.boxplot(scorer)

    # Extract statistical information from the boxplot
    minimum = boxplot['whiskers'][0].get_ydata()[1]
    maximum = boxplot['whiskers'][1].get_ydata()[1]
    q1 = boxplot['whiskers'][0].get_ydata()[0]
    q3 = boxplot['whiskers'][1].get_ydata()[0]
    iqr = q3 - q1

    # Set plot labels and title
    plt.title(f'{title} Distribution')
    plt.ylabel(f'{title}')

    # Show the boxplot
    plt.show()

    # Return the statistical values
    return (
        round(q1, 3),
        round(q3, 3),
        round(iqr, 3),
        round(minimum, 3),
        round(maximum, 3)
    )


def print_boxplot(q1, q3, iqr, minimum, maximum):
    print("Q1: ", q1)
    print("Q3: ", q3)
    print("IQR: ", iqr)
    print("Minimum: ", minimum)
    print("Maximum: ", maximum)


def plot_anomaly_imgs(anomaly_indexes, data, labels):
    # Plot some of the anomalies
    plt.figure(figsize=(12, 8))
    for i, idx in enumerate(anomaly_indexes[:10]):  # Plot the first 10 anomaly indices
        plt.subplot(2, 5, i + 1)
        plt.imshow(data[idx].reshape(28, 28), cmap='gray')
        plt.title(f'Label {labels[idx]}')
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


def scores_generator(model, data, batch_size):
    scores = []

    data_size = len(data)
    num_batches = (data_size + batch_size - 1) // batch_size

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        batch_data = data[start_idx:end_idx]

        batch_scores = model.decision_function(batch_data)
        scores.extend(batch_scores)

    return np.array(scores)