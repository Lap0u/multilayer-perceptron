# Setup plotting
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import GroupShuffleSplit

import keras
from keras import layers
from keras import callbacks
import ml_tools as ml
import numpy as np

# Set Matplotlib defaults
plt.rc("figure", autolayout=True)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=18,
    titlepad=10,
)
plt.rc("animation", html="html5")


def diagnosis_to_numeric(y):
    return y.apply(lambda x: 1 if x == "M" else 0)


def train():

    df = ml.load_csv("dataset/train.csv")
    validation_df = ml.load_csv("dataset/validation.csv")
    validation_y = validation_df.pop("Diagnosis")
    validation_y = np.array(diagnosis_to_numeric(validation_y), ndmin=2)
    validation_df = ml.normalize_df(validation_df)
    df.drop("ID", axis=1, inplace=True)
    validation_df.drop("ID", axis=1, inplace=True)
    y = df.pop("Diagnosis")
    y = np.array(diagnosis_to_numeric(y), ndmin=2)
    df = ml.normalize_df(df)

    model = keras.Sequential(
        [
            layers.Dense(24, activation="relu", input_shape=[df.shape[1]]),
            layers.Dense(24, activation="relu"),
            layers.Dense(24, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="mae",
    )
    history = model.fit(
        df,
        y,
        validation_data=(validation_df, validation_y),
        batch_size=512,
        epochs=50,
        verbose=0,
    )
    history_df = pd.DataFrame(history.history)
    history_df.loc[:, ["loss", "val_loss"]].plot()
    print("Minimum Validation Loss: {:0.4f}".format(history_df["val_loss"].min()))


train()
