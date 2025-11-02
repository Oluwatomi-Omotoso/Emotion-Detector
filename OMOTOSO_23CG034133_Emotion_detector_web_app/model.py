# Dependencies.
import os
import numpy as np
import pandas as pd
import tensorflow
from tensorflow import keras
from keras.models import Sequential
from keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization,
)
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint


# Paths
FER_CSV = "./fer2013.csv"


# This function's for loading the fer 2013 as a dataset.
def load_fer(csv_path):
    df = pd.read_csv(csv_path)
    X, y = [], []
    for idx, row in df.iterrows():
        pixels = np.fromstring(row["pixels"], sep=" ", dtype=np.float32)
        X.append(pixels.reshape(48, 48, 1) / 255.0)
        y.append(int(row["emotion"]))
    X = np.array(X)
    y = to_categorical(np.array(y), num_classes=7)
    return X, y


def build_model(input_shape=(48, 48, 1), num_classes=7):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation="relu", input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer=Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


if __name__ == "__main__":
    if not os.path.exists(FER_CSV):
        raise FileNotFoundError(
            "Download fer2013.csv and place it in project root. See https://www.kaggle.com/datasets/msambare/fer2013 or other mirrors."
        )

    X, y = load_fer(FER_CSV)
    # split small validation
    from sklearn.model_selection import train_test_split

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )

    model = build_model()
    os.makedirs("models", exist_ok=True)
    checkpoint = ModelCheckpoint(
        "models/emotion_cnn.h5", monitor="val_accuracy", save_best_only=True, verbose=1
    )
    model.fit(
        X_train,
        y_train,
        epochs=30,
        batch_size=64,
        validation_data=(X_val, y_val),
        callbacks=[checkpoint],
    )
