import matplotlib.pyplot as plt
import argparse
import ml_tools as ml
import numpy as np
from typing import Dict, Tuple
import copy

# TANH 0.1431  1e-15 0.9 0.99 / 24 24 24 (xavier normalized init (seed 42) loss 0.054)
# TANH 0.01  1e-15 0.9 0.99 / 24 24 24 (xavier normalized init (seed 42) loss 0.060) MOMENTUM

LEARNING_RATE = 0.1431
EPOCHS = 2000
EPSILON = 1e-15
BETA = 0.9
BETA2 = 0.99
PATIENCE = 100


def get_optimizer(optimizer):
    if optimizer == "momentum":
        return _update_momentum
    if optimizer == "rmsprop":
        return _update_rmsprop
    if optimizer == "adam":
        return _update_adam
    return _update_sgd


def get_activation(activation):
    if activation == "tanh":
        return ml.tanh_, ml.tanh_derivative
    if activation == "relu":
        return ml.relu_, ml.relu_derivative
    return ml.sigmoid_, ml.sigmoid_derivative


def xavier_init(dimensions):
    np.random.seed(42)
    parameters = {}
    layer_len = len(dimensions)
    for layer in range(1, layer_len):
        lower_bound = -(1.0 / np.sqrt(dimensions[layer - 1]))
        upper_bound = 1.0 / np.sqrt(dimensions[layer - 1])
        parameters["slope_" + str(layer)] = np.random.uniform(
            lower_bound, upper_bound, (dimensions[layer], dimensions[layer - 1])
        )
        parameters["intercept_" + str(layer)] = 0
    return parameters


def xavier_normalized_init(dimensions):
    np.random.seed(42)
    parameters = {}
    layer_len = len(dimensions)
    for layer in range(1, layer_len):
        lower_bound = -(6.0 / np.sqrt(dimensions[layer - 1] + dimensions[layer]))
        upper_bound = 6.0 / np.sqrt(dimensions[layer - 1] + dimensions[layer])
        parameters["slope_" + str(layer)] = np.random.uniform(
            lower_bound, upper_bound, (dimensions[layer], dimensions[layer - 1])
        )
        parameters["intercept_" + str(layer)] = 0
    return parameters


def he_init(dimensions):
    np.random.seed(42)
    parameters = {}
    layer_len = len(dimensions)
    for layer in range(1, layer_len):
        std = np.sqrt(2 / dimensions[layer_len - 1])
        parameters["slope_" + str(layer)] = np.random.normal(
            0, std, (dimensions[layer], dimensions[layer - 1])
        )
        parameters["intercept_" + str(layer)] = 0
    return parameters


def get_init_params(dimensions, init):
    if init == "xavier":
        return xavier_init(dimensions)
    if init == "he":
        return he_init(dimensions)
    if init == "nxavier":
        return xavier_normalized_init(dimensions)
    return xavier_normalized_init(dimensions)


def forward_propagation(X, parameters, activation_function):
    activations = {"Activation_0": X}
    layer_len = len(parameters) // 2
    for layer in range(1, layer_len + 1):
        layer_slope = parameters["slope_" + str(layer)]
        layer_intercept = parameters["intercept_" + str(layer)]
        prev_layer_activation = activations["Activation_" + str(layer - 1)]

        Z = np.dot(layer_slope, prev_layer_activation) + layer_intercept
        if layer == layer_len:
            curr_layer_activation = ml.softmax_(Z)
        else:
            curr_layer_activation = activation_function(Z)
        activations["Activation_" + str(layer)] = curr_layer_activation

    return activations


def log_loss(A, y):
    return (
        -1
        / len(y)
        * np.sum(y * np.log(A + EPSILON) + (1 - y) * np.log(1 - A + EPSILON))
    )


def back_propagation(activations, y, parameters, actxavier_initivation_derivative):
    layer_len = len(parameters) // 2
    m = y.shape[1]

    dZ = activations["Activation_" + str(layer_len)] - y
    gradients = {}
    for layer in reversed(range(1, layer_len + 1)):
        prev_activation = activations["Activation_" + str(layer - 1)]
        curr_slope = parameters["slope_" + str(layer)]
        gradients["d_slopes_" + str(layer)] = 1 / m * np.dot(dZ, prev_activation.T)
        gradients["d_intercept_" + str(layer)] = (
            1 / m * np.sum(dZ, axis=1, keepdims=True)
        )
        if layer > 1:
            dZ = activation_derivative(prev_activation) * np.dot(curr_slope.T, dZ)
    return gradients


def update_parameters(
    gradients: Dict[str, np.ndarray],
    parameters: Dict[str, np.ndarray],
    optimizer_func: str,
    change_slopes: Dict[str, np.ndarray],
    change_intercepts: Dict[str, np.ndarray],
    S_slopes: Dict[str, np.ndarray],
    S_intercepts: Dict[str, np.ndarray],
) -> Tuple[
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
]:
    """
    Update neural network parameters using the specified optimizer.

    Args:
        gradients: Dictionary containing gradients for slopes and intercepts.
        parameters: Dictionary containing current parameter values.
        optimizer: String specifying the optimization algorithm ('momentum', 'rmsprop', 'adam', or 'sgd').
        change_slopes: Dictionary containing momentum for slopes.
        change_intercepts: Dictionary containing momentum for intercepts.
        S_slopes: Dictionary containing squared gradient accumulations for slopes (used in RMSprop and Adam).
        S_intercepts: Dictionary containing squared gradient accumulations for intercepts (used in RMSprop and Adam).
    Returns:
        Tuple containing updated parameters and optimizer states.
    """
    layer_count = len(parameters) // 2

    for layer in range(1, layer_count + 1):
        slope_key = f"slope_{layer}"
        intercept_key = f"intercept_{layer}"
        d_slope_key = f"d_slopes_{layer}"
        d_intercept_key = f"d_intercept_{layer}"

        optimizer_func(
            parameters,
            gradients,
            slope_key,
            intercept_key,
            d_slope_key,
            d_intercept_key,
            change_slopes,
            change_intercepts,
            S_slopes,
            S_intercepts,
        )

    return parameters, change_slopes, change_intercepts, S_slopes, S_intercepts


def _update_momentum(
    parameters,
    gradients,
    slope_key,
    intercept_key,
    d_slope_key,
    d_intercept_key,
    change_slopes,
    change_intercepts,
    *args,
):
    change_slopes[slope_key] = (
        BETA * change_slopes[slope_key] - LEARNING_RATE * gradients[d_slope_key]
    )
    change_intercepts[intercept_key] = (
        BETA * change_intercepts[intercept_key]
        - LEARNING_RATE * gradients[d_intercept_key]
    )

    parameters[slope_key] += change_slopes[slope_key]
    parameters[intercept_key] += change_intercepts[intercept_key]


def _update_rmsprop(
    parameters,
    gradients,
    slope_key,
    intercept_key,
    d_slope_key,
    d_intercept_key,
    S_slopes,
    S_intercepts,
    *args,
):
    S_slopes[slope_key] = (
        BETA2 * S_slopes[slope_key] + (1 - BETA2) * gradients[d_slope_key] ** 2
    )
    S_intercepts[intercept_key] = (
        BETA2 * S_intercepts[intercept_key]
        + (1 - BETA2) * gradients[d_intercept_key] ** 2
    )

    parameters[slope_key] -= (
        LEARNING_RATE * gradients[d_slope_key] / np.sqrt(S_slopes[slope_key] + EPSILON)
    )
    parameters[intercept_key] -= (
        LEARNING_RATE
        * gradients[d_intercept_key]
        / np.sqrt(S_intercepts[intercept_key] + EPSILON)
    )


def _update_adam(
    parameters,
    gradients,
    slope_key,
    intercept_key,
    d_slope_key,
    d_intercept_key,
    change_slopes,
    change_intercepts,
    S_slopes,
    S_intercepts,
):
    change_slopes[slope_key] = (
        BETA * change_slopes[slope_key] + (1 - BETA) * gradients[d_slope_key]
    )
    change_intercepts[intercept_key] = (
        BETA * change_intercepts[intercept_key]
        + (1 - BETA) * gradients[d_intercept_key]
    )

    S_slopes[slope_key] = (
        BETA2 * S_slopes[slope_key] + (1 - BETA2) * gradients[d_slope_key] ** 2
    )
    S_intercepts[intercept_key] = (
        BETA2 * S_intercepts[intercept_key]
        + (1 - BETA2) * gradients[d_intercept_key] ** 2
    )

    parameters[slope_key] -= (
        LEARNING_RATE
        * change_slopes[slope_key]
        / np.sqrt(S_slopes[slope_key] + EPSILON)
    )
    parameters[intercept_key] -= (
        LEARNING_RATE
        * change_intercepts[intercept_key]
        / np.sqrt(S_intercepts[intercept_key] + EPSILON)
    )


def _update_sgd(
    parameters, gradients, slope_key, intercept_key, d_slope_key, d_intercept_key, *args
):
    parameters[slope_key] -= LEARNING_RATE * gradients[d_slope_key]
    parameters[intercept_key] -= LEARNING_RATE * gradients[d_intercept_key]


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
    return true_positives / (true_positives + false_negatives + EPSILON)


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
    return true_positives / (true_positives + false_positives + EPSILON)


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
    return loss, val_loss


def plot_f1(f1_history, val_f1_history):
    plt.plot(f1_history)
    plt.plot(val_f1_history)
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
    precision = true_positives / (true_positives + false_positives + EPSILON)
    recall = true_positives / (true_positives + false_negatives + EPSILON)
    return 2 * (precision * recall) / (precision + recall + EPSILON)


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


def print_best_loss_accuracy(loss_history, val_loss_history, accuracy, val_accuracy):
    # evite les cas ou la validation est plus basse que l'entrainement (training chanceux)
    masked_loss = [
        val_loss
        for val_loss, loss in zip(val_loss_history, loss_history)
        if loss <= val_loss
    ]

    best_epoch_index = np.argmin(masked_loss)
    best_epoch = next(
        i
        for i, (loss, val_loss) in enumerate(zip(loss_history, val_loss_history))
        if loss <= val_loss and val_loss == masked_loss[best_epoch_index]
    )
    print(f"Best Train Loss: {loss_history[best_epoch]:.3f}")
    print(f"Best Validation Loss: {val_loss_history[best_epoch]:.3f}")
    print(f"Best Train Accuracy: {accuracy[best_epoch]:.3f}")
    print(f"Best Validation Accuracy: {val_accuracy[best_epoch]:.3f}")
    print(f"Best Epoch: {best_epoch}")


def train_model(
    X,
    y,
    hidden_layer,
    validation_df,
    batch_size,
    validation_y,
    precision,
    recall,
    f1,
    optimizer,
    activation_function,
    activation_derivative,
    init,
    early_stopping=False,
):
    dimensions = list(hidden_layer)
    dimensions.insert(0, X.shape[0])
    dimensions.append(y.shape[0])
    parameters = get_init_params(dimensions, init)
    best_parameters = dict(parameters)
    layer_len = len(parameters) // 2
    best_loss = -1
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
    change_slopes = {"slope_" + str(layer): 0 for layer in range(1, layer_len + 1)}
    change_intercept = {
        "intercept_" + str(layer): 0 for layer in range(1, layer_len + 1)
    }
    S_slopes = {"slope_" + str(layer): 0 for layer in range(1, layer_len + 1)}
    S_intercept = {"intercept_" + str(layer): 0 for layer in range(1, layer_len + 1)}

    for i in range(EPOCHS):
        batched_X, batched_y = ml.get_mini_batches(X.T, y.T, batch_size)
        batched_y = batched_y.T
        batched_X = batched_X.T
        activation = forward_propagation(batched_X, parameters, activation_function)
        A_val = forward_propagation(validation_df, parameters, activation_function)
        gradients = back_propagation(
            activation, batched_y, parameters, activation_derivative
        )
        curr_train_loss, curr_val_loss = update_log_loss(
            activation,
            batched_y,
            A_val,
            validation_y,
            loss_history,
            val_loss_history,
            layer_len,
        )
        if curr_train_loss <= curr_val_loss and (
            curr_val_loss < best_loss or best_loss == -1
        ):
            best_parameters = copy.deepcopy(parameters)
            best_loss = curr_val_loss
        parameters, change_slopes, change_intercept, S_slopes, S_intercept = (
            update_parameters(
                gradients,
                parameters,
                optimizer,
                change_slopes,
                change_intercept,
                S_slopes,
                S_intercept,
            )
        )

        update_accuracy(
            activation,
            batched_y,
            A_val,
            validation_y,
            accuracy,
            val_accuracy,
            layer_len,
        )
        if recall:
            update_recall(
                activation,
                batched_y,
                A_val,
                validation_y,
                recall_history,
                val_recall_history,
                layer_len,
            )
        if precision:
            update_precision(
                activation,
                batched_y,
                A_val,
                validation_y,
                precision_history,
                val_precision_history,
                layer_len,
            )
        if f1:
            update_f1(
                activation,
                batched_y,
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
        if early_stopping and i > PATIENCE:
            if (
                val_loss_history[i] > val_loss_history[i - 1]
                and val_loss_history[i - 1] > val_loss_history[i - 2]
                and val_loss_history[i - 2] > val_loss_history[i - 3]
            ):
                break
    display_progress(loss_history, val_loss_history, accuracy, val_accuracy)
    if recall:
        plot_recall(recall_history, val_recall_history)
    if precision:
        plot_precision(precision_history, val_precision_history)
    if f1:
        plot_f1(f1_history, val_f1_history)
    print_best_loss_accuracy(loss_history, val_loss_history, accuracy, val_accuracy)
    return best_parameters


def try_predict(parameters, validation_df, validation_y, activation_function):
    # print("weights", parameters)
    print("activation_function", activation_function)
    A_val = forward_propagation(validation_df, parameters, activation_function)
    print("A_val", A_val["Activation_4"])
    loss = log_loss(A_val["Activation_4"].flatten(), validation_y.flatten())
    print("Loss for the given dataset with trained weights : ", loss)


def display_predictions(x, y, parameters, validation_df, validation_y, activation):
    layer_len = len(parameters) // 2
    predictions = forward_propagation(x, parameters, activation)
    predictions = np.where(predictions["Activation_" + str(layer_len)] > 0.5, 1, 0)
    ml.plot_confusion_matrix(
        y.flatten(), predictions.flatten(), ["B", "M"], title="Train Confusion Matrix"
    )
    val_predictions = forward_propagation(validation_df, parameters, activation)
    val_predictions = np.where(
        val_predictions["Activation_" + str(layer_len)] > 0.5, 1, 0
    )
    ml.plot_confusion_matrix(
        validation_y.flatten(),
        val_predictions.flatten(),
        ["B", "M"],
        title="Validation Confusion Matrix",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t", "--train", required=True, help="Path to the train data file"
    )
    parser.add_argument(
        "-v", "--validation", required=True, help="Path to the validation data file"
    )
    parser.add_argument("-r", "--recall", action="store_true", help="Plot the recall")
    parser.add_argument(
        "-p", "--precision", action="store_true", help="Plot the precision"
    )
    parser.add_argument("-f", "--f1", action="store_true", help="Plot the f1 score")
    parser.add_argument("-o", "--optimizer", help="Optimizer to use")
    parser.add_argument(
        "-a", "--activation", help="Activation function to use", default="sigmoid"
    )
    parser.add_argument(
        "-es", "--early-stopping", help="Early stopping", action="store_true"
    )
    parser.add_argument(
        "-hl",
        "--hidden-layers",
        nargs="+",
        help="Hidden layers",
        default=[24, 24, 24],
        type=int,
    )
    parser.add_argument("-i", "--init", help="Initialization method")
    parser.add_argument("-b", "--batch-size", help="Batch size", type=int)
    parser.add_argument(
        "-cm",
        "--confusion-matrix",
        action="store_true",
        help="Display confusion matrices",
    )
    args = parser.parse_args()
    try:
        activation, activation_derivative = get_activation(args.activation)
        optimizer = get_optimizer(args.optimizer)
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
    except Exception as e:
        print(e)
        exit(1)
    df = df.T
    validation_df = validation_df.T
    hidden_layers = args.hidden_layers
    parameters = train_model(
        X=df,
        y=y,
        hidden_layer=hidden_layers,
        validation_df=validation_df,
        validation_y=validation_y,
        batch_size=args.batch_size,
        precision=args.precision,
        recall=args.recall,
        f1=args.f1,
        optimizer=optimizer,
        activation_function=activation,
        activation_derivative=activation_derivative,
        early_stopping=args.early_stopping,
        init=args.init,
    )
    if args.confusion_matrix:
        display_predictions(df, y, parameters, validation_df, validation_y, activation)
    save_data = {
        "parameters": parameters,
        "activation": args.activation,
        "hidden_layers": args.hidden_layers,
    }
    np.save("parameters.npy", save_data)
