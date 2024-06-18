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


def print_metrics(i, loss_history, val_loss_history, accuracy, val_accuracy):
    print(f"Epoch: {i}")
    print(f"Train Loss: {loss_history[i]:.3f}")
    print(f"Validation Loss: {val_loss_history[i]:.3f}")
    print(f"Train Accuracy: {accuracy[i]:.3f}")
    print(f"Validation Accuracy: {val_accuracy[i]:.3f}")
    print()


def diagnosis_to_numeric(y):
    return y.apply(lambda x: 1 if x == "M" else 0)


def display_progress(loss_history, val_loss_history, accuracy, val_accuracy):
    plt.plot(loss_history)
    plt.plot(val_loss_history)
    plt.plot(accuracy)
    plt.plot(val_accuracy)
    plt.xscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.ylim(0, 1.2)
    plt.xlim(1, EPOCHS)
    plt.title("Loss and Accuracy vs Epoch")
    plt.legend(
        ["Train Loss", "Validation Loss", "Train Accuracy", "Validation Accuracy"]
    )
    plt.show()


def compute_accuracy(A, y):
    predictions = np.where(A > 0.5, 1, 0)
    return np.sum(predictions == y) / len(y)


def train_model(slopes, intercept, x, y, validation_df, validation_y):
    loss_history = []
    val_loss_history = []
    accuracy = []
    val_accuracy = []
    for i in range(EPOCHS):
        A = model(slopes, intercept, x)
        A_val = model(slopes, intercept, validation_df)
        loss_history.append(log_loss(A, y))
        val_loss_history.append(log_loss(A_val, validation_y))
        accuracy.append(compute_accuracy(A, y))
        val_accuracy.append(compute_accuracy(A_val, validation_y))
        dW, db = gradients(A, x, y)
        slopes, intercept = update(slopes, intercept, dW, db)
        if i % 1000 == 0:
            print_metrics(i, loss_history, val_loss_history, accuracy, val_accuracy)
    display_progress(loss_history, val_loss_history, accuracy, val_accuracy)
    return slopes, intercept


def display_predictions(x, y, slopes, intercept, validation_df, validation_y):
    predictions = model(slopes, intercept, x)
    predictions = np.where(predictions > 0.5, 1, 0)
    ml.plot_confusion_matrix(y, predictions, ["B", "M"], title="Train Confusion Matrix")
    val_predictions = model(slopes, intercept, validation_df)
    val_predictions = np.where(val_predictions > 0.5, 1, 0)
    ml.plot_confusion_matrix(
        validation_y, val_predictions, ["B", "M"], title="Validation Confusion Matrix"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train", help="Path to the train data file")
    parser.add_argument("-v", "--validation", help="Path to the validation data file")
    parser.add_argument(
        "-cm",
        "--confusion-matrix",
        action="store_true",
        help="Display confusion matrices",
    )
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
    slopes, intercept = train_model(
        slopes, intercept, df, y, validation_df, validation_y
    )
    if args.confusion_matrix:
        display_predictions(df, y, slopes, intercept, validation_df, validation_y)
