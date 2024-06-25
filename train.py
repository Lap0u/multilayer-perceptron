import matplotlib.pyplot as plt
import argparse
import ml_tools as ml
import numpy as np

LEARNING_RATE = 0.1
EPOCHS = 1700
EPSILON = 1e-15


def init(dimensions):
    parametres = {}
    layer_len = len(dimensions)
    np.random.seed(1)

    for layer in range(1, layer_len):
        parametres["slope_" + str(layer)] = np.random.randn(
            dimensions[layer], dimensions[layer - 1]
        )
        parametres["intercept_" + str(layer)] = np.random.randn(dimensions[layer], 1)
    return parametres


def forward_propagation(X, parametres):
    activations = {"Activation_0": X}
    layer_len = len(parametres) // 2
    for layer in range(1, layer_len + 1):
        layer_slope = parametres["slope_" + str(layer)]
        layer_intercept = parametres["intercept_" + str(layer)]
        prev_layer_activation = activations["Activation_" + str(layer - 1)]

        Z = np.dot(layer_slope, prev_layer_activation) + layer_intercept
        curr_layer_activation = 1 / (1 + np.exp(-Z))
        activations["Activation_" + str(layer)] = curr_layer_activation

    return activations


def log_loss(A, y):
    return (
        -1
        / len(y)
        * np.sum(y * np.log(A + EPSILON) + (1 - y) * np.log(1 - A + EPSILON))
    )


def back_propagation(activations, y, parametres):
    layer_len = len(parametres) // 2
    m = y.shape[1]

    dZ = activations["Activation_" + str(layer_len)] - y
    gradients = {}
    for layer in reversed(range(1, layer_len + 1)):
        prev_activation = activations["Activation_" + str(layer - 1)]
        curr_slope = parametres["slope_" + str(layer)]
        gradients["d_slopes_" + str(layer)] = 1 / m * np.dot(dZ, prev_activation.T)
        gradients["d_intercept_" + str(layer)] = (
            1 / m * np.sum(dZ, axis=1, keepdims=True)
        )
        if layer > 1:
            dZ = np.dot(curr_slope.T, dZ) * (prev_activation * (1 - prev_activation))
    return gradients


def update(gradients, parametres):
    layer_len = len(parametres) // 2
    for layer in range(1, layer_len + 1):
        parametres["slope_" + str(layer)] = (
            parametres["slope_" + str(layer)]
            - LEARNING_RATE * gradients["d_slopes_" + str(layer)]
        )
        parametres["intercept_" + str(layer)] = (
            parametres["intercept_" + str(layer)]
            - LEARNING_RATE * gradients["d_intercept_" + str(layer)]
        )
    return parametres


def print_metrics(
    i,
    loss_history,
    val_loss_history,
    accuracy,
    val_accuracy,
    recall_history,
    val_recall_history,
    precision_history,
    val_precision_history,
    f1_history,
    val_f1_history,
):
    print(f"Epoch: {i}")
    print(f"Train Loss: {loss_history[i]:.3f}")
    print(f"Validation Loss: {val_loss_history[i]:.3f}")
    print(f"Train Accuracy: {accuracy[i]:.3f}")
    print(f"Validation Accuracy: {val_accuracy[i]:.3f}")
    if recall_history:
        print(f"Train Recall: {recall_history[i]:.3f}")
        print(f"Validation Recall: {val_recall_history[i]:.3f}")
    if precision_history:
        print(f"Train Precision: {precision_history[i]:.3f}")
        print(f"Validation Precision: {val_precision_history[i]:.3f}")
    if f1_history:
        print(f"Train F1: {f1_history[i]:.3f}")
        print(f"Validation F1: {val_f1_history[i]:.3f}")
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


def plot_precision(precision_history, val_precision_history):
    plt.plot(precision_history)
    plt.plot(val_precision_history)
    plt.xscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Precision")
    plt.ylim(0, 1.2)
    plt.xlim(1, EPOCHS)
    plt.title("Precision vs Epoch")
    plt.legend(["Train Precision", "Validation Precision"])
    plt.show()


def plot_recall(recall_history, val_recall_history):
    plt.plot(recall_history)
    plt.plot(val_recall_history)
    plt.xscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Recall")
    plt.ylim(0, 1.2)
    plt.xlim(1, EPOCHS)
    plt.title("Recall vs Epoch")
    plt.legend(["Train Recall", "Validation Recall"])
    plt.show()


def compute_accuracy(A, y):
    predictions = np.where(A > 0.5, 1, 0)
    return np.sum(predictions == y) / len(y)


def update_accuracy(
    activation, y, A_val, validation_y, accuracy, val_accuracy, layer_len
):
    acc = compute_accuracy(
        activation["Activation_" + str(layer_len)].flatten(), y.flatten()
    )
    val_acc = compute_accuracy(
        A_val["Activation_" + str(layer_len)].flatten(), validation_y.flatten()
    )
    accuracy.append(acc)
    val_accuracy.append(val_acc)


def compute_recall(A, y):
    A = np.where(A > 0.5, 1, 0)
    true_positives = np.sum(np.logical_and(A == 1, y == 1))
    false_negatives = np.sum(np.logical_and(A == 0, y == 1))
    return true_positives / (true_positives + false_negatives)


def update_recall(
    activation, y, A_val, validation_y, recall_history, val_recall_history, layer_len
):
    recall = compute_recall(
        activation["Activation_" + str(layer_len)].flatten(), y.flatten()
    )
    val_recall = compute_recall(
        A_val["Activation_" + str(layer_len)].flatten(), validation_y.flatten()
    )
    recall_history.append(recall)
    val_recall_history.append(val_recall)


def compute_precision(A, y):
    A = np.where(A > 0.5, 1, 0)
    true_positives = np.sum(np.logical_and(A == 1, y == 1))
    false_positives = np.sum(np.logical_and(A == 1, y == 0))
    return true_positives / (true_positives + false_positives)


def update_precision(
    activation,
    y,
    A_val,
    validation_y,
    precision_history,
    val_precision_history,
    layer_len,
):
    precision = compute_precision(
        activation["Activation_" + str(layer_len)].flatten(), y.flatten()
    )
    val_precision = compute_precision(
        A_val["Activation_" + str(layer_len)].flatten(), validation_y.flatten()
    )
    precision_history.append(precision)
    val_precision_history.append(val_precision)


def update_log_loss(
    activation, y, A_val, validation_y, loss_history, val_loss_history, layer_len
):
    loss = log_loss(activation["Activation_" + str(layer_len)].flatten(), y.flatten())
    val_loss = log_loss(
        A_val["Activation_" + str(layer_len)].flatten(), validation_y.flatten()
    )
    loss_history.append(loss)
    val_loss_history.append(val_loss)


def plot_f1(f1_history, val_f1_history):
    plt.plot(f1_history)
    plt.plot(val_f1_history)
    plt.xscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.ylim(0, 1.2)
    plt.xlim(1, EPOCHS)
    plt.title("F1 Score vs Epoch")
    plt.legend(["Train F1 Score", "Validation F1 Score"])
    plt.show()


def compute_f1(A, y):
    A = np.where(A > 0.5, 1, 0)
    true_positives = np.sum(np.logical_and(A == 1, y == 1))
    false_positives = np.sum(np.logical_and(A == 1, y == 0))
    false_negatives = np.sum(np.logical_and(A == 0, y == 1))
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    return 2 * (precision * recall) / (precision + recall)


def update_f1(
    activation,
    y,
    A_val,
    validation_y,
    f1_history,
    val_f1_history,
    layer_len,
):

    f1 = compute_f1(activation["Activation_" + str(layer_len)].flatten(), y.flatten())
    val_f1 = compute_f1(
        A_val["Activation_" + str(layer_len)].flatten(), validation_y.flatten()
    )
    f1_history.append(f1)
    val_f1_history.append(val_f1)


def train_model(X, y, hidden_layer, validation_df, validation_y, precision, recall, f1):
    dimensions = list(hidden_layer)
    dimensions.insert(0, X.shape[0])
    dimensions.append(y.shape[0])
    parametres = init(dimensions)
    layer_len = len(parametres) // 2
    loss_history = []
    val_loss_history = []
    accuracy = []
    val_accuracy = []
    precision_history = []
    val_precision_history = []
    recall_history = []
    val_recall_history = []
    f1_history = []
    val_f1_history = []
    activations = forward_propagation(X, parametres)
    gradients = back_propagation(activations, y, parametres)
    for i in range(EPOCHS):
        activation = forward_propagation(X, parametres)
        A_val = forward_propagation(validation_df, parametres)
        gradients = back_propagation(activation, y, parametres)
        parametres = update(gradients, parametres)
        update_log_loss(
            activation,
            y,
            A_val,
            validation_y,
            loss_history,
            val_loss_history,
            layer_len,
        )
        update_accuracy(
            activation, y, A_val, validation_y, accuracy, val_accuracy, layer_len
        )
        if recall:
            update_recall(
                activation,
                y,
                A_val,
                validation_y,
                recall_history,
                val_recall_history,
                layer_len,
            )
        if precision:
            update_precision(
                activation,
                y,
                A_val,
                validation_y,
                precision_history,
                val_precision_history,
                layer_len,
            )
        if f1:
            update_f1(
                activation,
                y,
                A_val,
                validation_y,
                f1_history,
                val_f1_history,
                layer_len,
            )
        if i % 10 == 0:
            print_metrics(
                i,
                loss_history,
                val_loss_history,
                accuracy,
                val_accuracy,
                recall_history,
                val_recall_history,
                precision_history,
                val_precision_history,
                f1_history,
                val_f1_history,
            )
    display_progress(loss_history, val_loss_history, accuracy, val_accuracy)
    if recall:
        plot_recall(recall_history, val_recall_history)
    if precision:
        plot_precision(precision_history, val_precision_history)
    if f1:
        plot_f1(f1_history, val_f1_history)
    return slopes, intercept


def display_predictions(x, y, slopes, intercept, validation_df, validation_y):
    predictions = forward_propagation(slopes, intercept, x)
    predictions = np.where(predictions > 0.5, 1, 0)
    ml.plot_confusion_matrix(y, predictions, ["B", "M"], title="Train Confusion Matrix")
    val_predictions = forward_propagation(slopes, intercept, validation_df)
    val_predictions = np.where(val_predictions > 0.5, 1, 0)
    ml.plot_confusion_matrix(
        validation_y, val_predictions, ["B", "M"], title="Validation Confusion Matrix"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train", help="Path to the train data file")
    parser.add_argument("-v", "--validation", help="Path to the validation data file")
    parser.add_argument("-r", "--recall", action="store_true", help="Plot the recall")
    parser.add_argument(
        "-p", "--precision", action="store_true", help="Plot the precision"
    )
    parser.add_argument("-f", "--f1", action="store_true", help="Plot the f1 score")
    parser.add_argument(
        "-hl",
        "--hidden-layers",
        nargs="+",
        help="Hidden layers",
        default=[24, 24, 24],
        type=int,
    )
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
        validation_y = np.array(diagnosis_to_numeric(validation_y), ndmin=2)
        validation_df = ml.normalize_df(validation_df)
        df.drop("ID", axis=1, inplace=True)
        validation_df.drop("ID", axis=1, inplace=True)
        y = df.pop("Diagnosis")
        y = np.array(diagnosis_to_numeric(y), ndmin=2)
        df = ml.normalize_df(df)
        print(df)
    except Exception as e:
        print(e)
        exit(1)
    df = df.T
    validation_df = validation_df.T
    hidden_layers = args.hidden_layers
    slopes, intercept = train_model(
        df,
        y,
        hidden_layers,
        validation_df,
        validation_y,
        args.precision,
        args.recall,
        args.f1,
    )
    if args.confusion_matrix:
        display_predictions(df, y, slopes, intercept, validation_df, validation_y)
