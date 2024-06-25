import os
from os import path
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns
import math
import pandas as pd
from sklearn.metrics import confusion_matrix


def get_mini_batches(x, transformed_y, batch=None):
    if batch is None:
        return x, transformed_y
    rand_val = np.random.randint(0, len(x), batch)
    batched_x = x.iloc[rand_val]
    batched_y = transformed_y[rand_val]
    return batched_x, batched_y


def is_valid_path(file_path):
    if path.isfile(file_path) is False:
        raise Exception("File does not exist")
    if (os.access(file_path, os.R_OK)) is False:
        raise Exception("File is not readable")
    if Path(file_path).suffix != ".csv":
        raise Exception("File is not a csv file")


def plot_confusion_matrix(houses, house_predictions, labels, title="Confusion Matrix"):
    cm = confusion_matrix(houses, house_predictions)
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="coolwarm", xticklabels=labels, yticklabels=labels
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.show()


def heatmap(df):
    plt.figure(figsize=(48, 48))
    plt.title("Correlation Heatmap")
    plt.margins(1)
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    plt.show()


def normalize_array(x):
    return (x - x.min()) / (x.max() - x.min())


def normalize_df(df):
    for column in df.columns:
        if df[column].dtype != "object":
            df[column] = normalize_array(df[column])
    return df


def denormalize_array(list, elem):
    return (elem * (max(list) - min(list))) + min(list)


def sigmoid_(z):
    return 1 / (1 + np.exp(-z))


def load_csv(filename, header="infer"):
    return pd.read_csv(filename, sep=",", header=header)


def split_data(df, test_size):
    actual_size = math.floor(test_size * df.shape[0])
    train_set = df.iloc[actual_size:, :]
    test_set = df.iloc[:actual_size, :]
    return train_set, test_set


def save_csv(df, filename):
    df.to_csv(filename, index=False)


def clean_data(df):
    df = df.fillna(df.mean())
    return df
