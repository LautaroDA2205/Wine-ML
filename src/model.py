import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.functions import load_wine_data, split_data


def train_model(path: str = "data/wine.data"):
    df = load_wine_data(path)

    X_train, X_test, y_train, y_test = split_data(df)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    return model, scaler, X_train.columns