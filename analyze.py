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
    metrics = ["M", "SE", "W"]  # Mean, Standard Error, Worst

    headers = ["ID", "Diagnosis"]

    for metric in metrics:
        for feature in base_features:
            headers.append(f"{metric} {feature.capitalize()}")

    data.columns = headers
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", help="Path to the data file")
    parser.add_argument("-hm", "--heatmap", action="store_true", help="plot heatmap")
    args = parser.parse_args()
    try:
        ml.is_valid_path(args.path)
        df = ml.load_csv(args.path, header=None)
        df = generate_headers(df)
        print(df.describe())
        df.drop(columns=["Diagnosis"], inplace=True)
        df = ml.clean_data(df)

    except Exception as e:
        print(e)
        exit(1)
    if args.heatmap:
        ml.heatmap(df)
