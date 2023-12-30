# utility
import os

# data processing
import pandas as pd
import numpy as np

# tensorflow
import tensorflow as tf
from tensorflow import keras

# sklearn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, classification_report,
    precision_recall_curve, auc, confusion_matrix, roc_auc_score, roc_curve, f1_score
)
from sklearn.model_selection import train_test_split

# visualization
import matplotlib.pyplot as plt
import seaborn as sns

# misc
import json
from PIL import Image, ImageDraw, ImageFont
from sklearn.utils import shuffle
import time
import psutil
import threading


def load_data():
    mnist = keras.datasets.mnist
    anomalies = keras.datasets.fashion_mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    data = np.concatenate((x_train, x_test))
    labels = np.concatenate((y_train, y_test))
    
    # Store the original labels before conversion to binary
    original_labels = np.concatenate((y_test, y_train))

    return data, labels, original_labels


def preprocess_data(data, labels, use_validation=True):
    anomayly_indexes = [688, 420, 3064, 7500]
    if use_validation:
        train_data, val_data, train_labels, val_labels = train_test_split(data, labels, test_size=0.2, random_state=42)
        train_data, _, train_labels, _ = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)

        test_data, test_labels = get_testset()

        train_data = train_data.astype('float32') / 255.0
        val_data = val_data.astype('float32') / 255.0
        test_data = test_data.astype('float32') / 255.0

        train_data = train_data.reshape(-1, 28 * 28)
        val_data = val_data.reshape(-1, 28 * 28)
        test_data = test_data.reshape(-1, 28 * 28)

        # Create true_labels for test set
        anom_data = test_data[anomayly_indexes]
        test_true_labels = np.ones(len(test_labels), dtype=bool)
        test_true_labels[anomayly_indexes] = False
        
        return train_data, train_labels, test_data, test_labels, val_data, val_labels, test_true_labels, anom_data
    else:
        train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

        test_data, test_labels = get_testset()

        train_data = train_data.astype('float32') / 255.0
        test_data = test_data.astype('float32') / 255.0

        train_data = train_data.reshape(-1, 28 * 28)
        test_data = test_data.reshape(-1, 28 * 28)

        # Create true_labels for test set
        anom_data = test_data[anomayly_indexes]
        anom_labels = test_labels[anomayly_indexes]
        test_true_labels = np.ones(len(test_labels), dtype=bool)
        test_true_labels[anomayly_indexes] = False

        return train_data, train_labels, test_data, test_labels, test_true_labels, anom_data, anom_labels
    

def get_testset():
    with open('../test_X.npy', 'rb') as f:
        test_data = np.load(f)

    with open('../test_Y.npy', 'rb') as f:
        test_labels = np.load(f)

    return test_data, test_labels


def plot_normalized_pixel_data(normal_data, anomaly_data):
    plt.grid()
    plt.plot(np.arange(784), normal_data[-1])
    plt.title("Normal data")
    plt.show()

    plt.grid()
    plt.plot(np.arange(784), anomaly_data[-1])
    plt.title("Abnormal data")
    plt.show()


def predict(model, data, threshold):
    reconstructions = model(data)
    loss = tf.keras.losses.mse(reconstructions, data)
    proba = np.mean(np.square(data - reconstructions), axis=1)
    return tf.math.less(loss, threshold), loss, proba


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


# TODO: Remove
# Function to generate anomalies (letters) and labels
def generate_letter_anomalies(num_anomalies=100, font_path="Arial"):
    font_size = 28  # Adjust font size as needed
    font = ImageFont.truetype(font_path, font_size)
    
    anomalies = []
    labels = []

    for _ in range(num_anomalies):
        # Choose a random letter from the alphabet
        letter = chr(np.random.randint(65, 91))  # ASCII codes for uppercase letters
        
        # Create a blank image
        img = Image.new('L', (28, 28), color=0)
        draw = ImageDraw.Draw(img)

        # Draw the letter on the image
        draw.text((0, 0), letter, fill=255, font=font)

        # Convert the image to a NumPy array and normalize
        img_array = np.array(img) / 255.0

        # Append the anomaly to the list
        anomalies.append(img_array)
        labels.append(0)  # Anomaly label

    return np.array(anomalies), np.array(labels)

# Function to concatenate anomalies with test data and shuffle
def concatenate_and_shuffle(test_data, test_labels, anomalies, anomaly_labels):
    concatenated_data = np.concatenate((test_data, anomalies))
    concatenated_labels = np.concatenate((test_labels, anomaly_labels))
    
    # Shuffle the data and labels together
    shuffled_data, shuffled_labels = shuffle(concatenated_data, concatenated_labels, random_state=42)
    
    return shuffled_data, shuffled_labels

# Function to save shuffled data and labels to .npy files
def save_to_npy(data, labels, data_filename="shuffled_data.npy", labels_filename="shuffled_labels.npy"):
    np.save(data_filename, data)
    np.save(labels_filename, labels)
    print(f"Data saved to {data_filename}")
    print(f"Labels saved to {labels_filename}")
