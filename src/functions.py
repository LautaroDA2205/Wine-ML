import pandas as pd

def load_wine_data(path):
    """
    Carga el dataset de vinos y asigna nombres a las columnas.
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

    # Ajustamos clases a 0,1,2
    df["class"] = df["class"] - 1

    return df


def class_summary(df):
    """
    Devuelve estadísticas descriptivas agrupadas por clase.
    """
    if "class" not in df.columns:
        raise ValueError("El DataFrame debe contener la columna 'class'.")

    return df.groupby("class").agg(["mean", "std"])