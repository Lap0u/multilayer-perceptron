import argparse
import numpy as np
import ml_tools as ml
import train


def predict(df, parameters):
    activation_function, _ = train.get_activation(parameters["activation"])
    weights = parameters["parameters"]
    predictions = train.forward_propagation(df, weights, activation_function)
    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="Path to the dataset"
    )
    parser.add_argument(
        "-p", "--parameters", type=str, required=True, help="Path to the parameters"
    )
    args = parser.parse_args()
    try:
        ml.is_valid_path(args.dataset)
        df = ml.load_csv(args.dataset)
        parameters = np.load(args.parameters, allow_pickle=True)
        parameters = parameters.item()
        df.drop("ID", axis=1, inplace=True)
        y = df.pop("Diagnosis")
        y = np.array(train.diagnosis_to_numeric(y), ndmin=2)
        df = ml.normalize_df(df)
        df = df.T
        y = y.T
        layer_len = len(parameters["parameters"]) // 2
    except Exception as e:
        print(e)
        exit(1)
    activation = predict(df, parameters)
    predictions = activation["Activation_" + str(layer_len)]
    print(predictions)
    loss = train.log_loss(predictions.flatten(), y.flatten())
    print("Loss for the given dataset with trained weights : ", loss)
    predictions = np.where(predictions > 0.5, 1, 0)
    print("predictions : ", predictions)
    ml.plot_confusion_matrix(
        predictions.flatten(), y.flatten(), ["B", "M"], title="Confusion Matrix"
    )
