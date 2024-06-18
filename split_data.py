import ml_tools as ml
import argparse


def generate_headers(data):
    base_features = [
        "radius",
        "texture",
        "perimeter",
        "area",
        "smoothness",
        "compactness",
        "concavity",
        "concave points",
        "symmetry",
        "fractal dimension",
    ]
    metrics = ["Mean", "SE", "Worst"]

    headers = ["ID", "Diagnosis"]

    for metric in metrics:
        for feature in base_features:
            headers.append(f"{metric} {feature.capitalize()}")

    data.columns = headers
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", help="Path to the data file")
    args = parser.parse_args()
    try:
        ml.is_valid_path(args.path)
        data = ml.load_csv(args.path)
        data = generate_headers(data)
        train, validation = ml.split_data(data, test_size=0.2)
        ml.save_csv(train, "dataset/train.csv")
        ml.save_csv(validation, "dataset/validation.csv")
    except Exception as e:
        print(e)
        exit(1)
