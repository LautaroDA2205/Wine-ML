import pandas as pd
from sklearn.model_selection import train_test_split


def load_wine_data(path: str) -> pd.DataFrame:
    """
    Carga el dataset de vinos y asigna nombres a las columnas.
    Ajusta las clases a valores 0, 1 y 2.
    """

    columns = [
        "class", "alcohol", "malic_acid", "ash", "alcalinity_of_ash",
        "magnesium", "total_phenols", "flavanoids",
        "nonflavanoid_phenols", "proanthocyanins",
        "color_intensity", "hue",
        "od280_od315", "proline"
    ]

    df = pd.read_csv(path, header=None)
    df.columns = columns

    # Ajustamos clases a 0, 1 y 2
    df["class"] = df["class"] - 1

    return df


def class_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Devuelve estadísticas descriptivas agrupadas por clase.
    """

    if "class" not in df.columns:
        raise ValueError("El DataFrame debe contener la columna 'class'.")

    return df.groupby("class").agg(["mean", "std"])


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Divide el dataset en entrenamiento y prueba
    utilizando estratificación.
    """

    X = df.drop("class", axis=1)
    y = df["class"]

    return train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )