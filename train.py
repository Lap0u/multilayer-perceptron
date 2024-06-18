import matplotlib.pyplot as plt
import argparse
import ml_tools as ml
import numpy as np

LEARNING_RATE = 0.1
EPOCHS = 10000
EPSILON = 1e-15


def init(df):
    slopes = np.random.rand(df.shape[1])
    intercept = np.random.rand(1)
    return slopes, intercept


def model(slopes, intercept, x):
    Z = np.dot(x, slopes) + intercept
    A = 1 / (1 + np.exp(-Z))
    return A


def log_loss(A, y):
    return (
        1
        / len(y)
        * np.sum(-y * np.log(A + EPSILON) - (1 - y) * np.log(1 - A + EPSILON))
    )


def gradients(A, X, y):
    dW = 1 / len(y) * np.dot(X.T, (A - y))
    db = 1 / len(y) * np.sum(A - y)
    return dW, db


def update(slopes, intercept, dW, db):
    slopes = slopes - LEARNING_RATE * dW
    intercept = intercept - LEARNING_RATE * db
    return slopes, intercept


def diagnosis_to_numeric(y):
    return y.apply(lambda x: 1 if x == "M" else 0)


def display_loss(loss_history):
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs Epoch")
    plt.show()


def train_model(slopes, intercept, x, y):
    loss_history = []
    for i in range(EPOCHS):
        A = model(slopes, intercept, x)
        loss_history.append(log_loss(A, y))
        dW, db = gradients(A, x, y)
        slopes, intercept = update(slopes, intercept, dW, db)
        print(f"Epoch: {i}, Loss: {loss_history[i]}")
    display_loss(loss_history)
    return slopes, intercept


def display_predictions(x, y, slopes, intercept):
    predictions = model(slopes, intercept, x)
    predictions = np.where(predictions > 0.5, 1, 0)
    ml.plot_confusion_matrix(y, predictions, ["B", "M"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train", help="Path to the train data file")
    parser.add_argument("-v", "--validation", help="Path to the validation data file")
    args = parser.parse_args()
    try:
        ml.is_valid_path(args.train)
        ml.is_valid_path(args.validation)
        df = ml.load_csv(args.train)
        validation_df = ml.load_csv(args.validation)
        validation_y = validation_df.pop("Diagnosis")
        validation_y = diagnosis_to_numeric(validation_y)
        validation_df = ml.normalize_df(validation_df)
        y = df.pop("Diagnosis")
        y = diagnosis_to_numeric(y)
        df = ml.normalize_df(df)
    except Exception as e:
        print(e)
        exit(1)
    slopes, intercept = init(df)
    slopes, intercept = train_model(slopes, intercept, df, y)
    display_predictions(df, y, slopes, intercept)
