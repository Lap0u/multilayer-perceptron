import matplotlib.pyplot as plt
import argparse
import ml_tools as ml
import numpy as np

LEARNING_RATE = 0.02
EPOCHS = 10000
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
        1
        / len(y)
        * np.sum(-y * np.log(A + EPSILON) - (1 - y) * np.log(1 - A + EPSILON))
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


def gradients(A, X, y):
    dW = 1 / len(y) * np.dot(X.T, (A - y))
    db = 1 / len(y) * np.sum(A - y)
    return dW, db


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
    # plt.xscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    # plt.ylim(0, 1.2)
    plt.xlim(1, EPOCHS)
    plt.title("Loss and Accuracy vs Epoch")
    plt.legend(
        ["Train Loss", "Validation Loss", "Train Accuracy", "Validation Accuracy"]
    )
    plt.show()


def compute_accuracy(A, y):
    predictions = np.where(A > 0.5, 1, 0)
    return np.sum(predictions == y) / len(y)


def train_model(X, y, hidden_layer, validation_df, validation_y):
    dimensions = list(hidden_layer)
    dimensions.insert(0, X.shape[0])
    dimensions.append(y.shape[0])
    parametres = init(dimensions)
    loss_history = []
    val_loss_history = []
    accuracy = []
    val_accuracy = []
    activations = forward_propagation(X, parametres)
    gradients = back_propagation(activations, y, parametres)
    for i in range(EPOCHS):
        activation = forward_propagation(X, parametres)
        A_val = forward_propagation(validation_df, parametres)
        gradients = back_propagation(activation, y, parametres)
        parametres = update(gradients, parametres)
        # if i % 10 == 0:
        layer_len = len(parametres) // 2
        loss = log_loss(
            activation["Activation_" + str(layer_len)].flatten(), y.flatten()
        )
        val_loss = log_loss(
            A_val["Activation_" + str(layer_len)].flatten(), validation_y.flatten()
        )
        # print(loss, i, "loss")
        loss_history.append(loss)
        val_loss_history.append(val_loss)
        acc = compute_accuracy(
            activation["Activation_" + str(layer_len)].flatten(), y.flatten()
        )
        # print(acc, i, "acc")
        val_acc = compute_accuracy(
            A_val["Activation_" + str(layer_len)].flatten(), validation_y.flatten()
        )
        accuracy.append(acc)
        val_accuracy.append(val_acc)
        if i % 100 == 0:
            print_metrics(i, loss_history, val_loss_history, accuracy, val_accuracy)
    display_progress(loss_history, val_loss_history, accuracy, val_accuracy)
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
    hidden_layers = [24, 24, 24]
    slopes, intercept = train_model(df, y, hidden_layers, validation_df, validation_y)
    if args.confusion_matrix:
        display_predictions(df, y, slopes, intercept, validation_df, validation_y)
