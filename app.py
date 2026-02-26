# ==================================================
# Wine Classification Simulator
# ==================================================

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# --------------------------------------------------
# Page Config
# --------------------------------------------------

st.set_page_config(
    page_title="Wine Classification App",
    page_icon="🍷",
    layout="wide"
)

st.title("🍷 Wine Classification Simulator")

st.markdown("""
Aplicación interactiva para simular la clasificación de vinos 
utilizando un modelo de **Logistic Regression** validado mediante 
Cross Validation (5 folds).

El modelo fue seleccionado por su estabilidad, robustez e interpretabilidad.
""")


# --------------------------------------------------
# Load Data
# --------------------------------------------------

@st.cache_resource
def load_data():
    df = pd.read_csv("data/wine.data", header=None)

    df.columns = [
        "class", "alcohol", "malic_acid", "ash", "alcalinity_of_ash",
        "magnesium", "total_phenols", "flavanoids",
        "nonflavanoid_phenols", "proanthocyanins",
        "color_intensity", "hue", "od280_od315", "proline"
    ]

    return df


# --------------------------------------------------
# Train Model
# --------------------------------------------------

@st.cache_resource
def train_model(df):
    X = df.drop("class", axis=1)
    y = df["class"]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ])

    model.fit(X, y)
    return model, X.columns


df = load_data()
model, feature_names = train_model(df)


# --------------------------------------------------
# Model Overview
# --------------------------------------------------

st.subheader("Model Overview")

st.markdown("""
- Modelo: Logistic Regression  
- Validación: Cross Validation (5 folds)  
- Métrica principal: Accuracy  
- Selección basada en estabilidad y menor variabilidad entre folds  
""")


# --------------------------------------------------
# Random Wine Generator
# --------------------------------------------------

def get_random_wine(df):
    random_row = df.sample(1).drop("class", axis=1)
    return random_row.iloc[0].to_dict()


if "wine_values" not in st.session_state:
    st.session_state.wine_values = {}

if st.button("🎲 Random Realistic Wine"):
    st.session_state.wine_values = get_random_wine(df)


# --------------------------------------------------
# Wine Simulator (Grouped Sliders)
# --------------------------------------------------

st.subheader("Wine Feature Simulator")

X = df.drop("class", axis=1)

def slider_group(features, title):
    with st.expander(title, expanded=False):
        for feature in features:
            min_val = float(X[feature].min())
            max_val = float(X[feature].max())
            mean_val = float(X[feature].mean())

            default_value = st.session_state.wine_values.get(feature, mean_val)

            st.session_state.wine_values[feature] = st.slider(
                feature,
                min_value=min_val,
                max_value=max_val,
                value=float(default_value)
            )


# Groups

slider_group(
    ["alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium"],
    "🍷 Composition"
)

slider_group(
    ["total_phenols", "flavanoids", "nonflavanoid_phenols", "proanthocyanins"],
    "🌿 Phenolic Content"
)

slider_group(
    ["color_intensity", "hue", "od280_od315", "proline"],
    "🎨 Visual & Chemical Profile"
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.session_state.wine_values:

    input_df = pd.DataFrame([st.session_state.wine_values])

    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]

    st.subheader("Prediction Result")

    st.success(f"Predicted Wine Class: {prediction}")

    wine_class_descriptions = {
    1: "Class 1 – Perfil equilibrado, buena estructura fenólica y balance general.",
    2: "Class 2 – Perfil más intenso, mayor color intensity y carácter más marcado.",
    3: "Class 3 – Perfil estructurado, alto contenido de flavanoids y proline."
}

    st.markdown(f"**Interpretation:** {wine_class_descriptions[prediction]}")

    proba_df = pd.DataFrame({
        "Class": model.named_steps["model"].classes_,
        "Probability": probabilities
    })

    st.bar_chart(proba_df.set_index("Class"))


# --------------------------------------------------
# How to Use
# --------------------------------------------------

st.markdown("---")
st.subheader("How to Use")

st.markdown("""
1. Ajusta los valores químicos utilizando los controles deslizantes.  
2. Alternativamente, genera un vino realista con el botón *Random Realistic Wine*.  
3. Observa la clase predicha y las probabilidades asociadas.  
4. Interpreta el perfil enológico asociado a cada clase.  

La simulación permite explorar cómo pequeñas variaciones químicas pueden 
afectar la clasificación del vino según el modelo entrenado.
""")