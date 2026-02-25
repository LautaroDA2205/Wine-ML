import streamlit as st
import pandas as pd
import sys

sys.path.append("src")

from src.model import train_model

st.set_page_config(page_title="Wine Classifier", layout="wide")

st.title("Wine Classification App")

@st.cache_resource
def load_trained_model():
    return train_model()

model, scaler, feature_names = load_trained_model()

st.sidebar.header("Input Features")

input_data = {}

for feature in feature_names:
    input_data[feature] = st.sidebar.number_input(feature, value=0.0)

input_df = pd.DataFrame([input_data])

scaled_input = scaler.transform(input_df)
prediction = model.predict(scaled_input)
proba = model.predict_proba(scaled_input)

st.subheader("Prediction")
st.write(f"Predicted Class: {prediction[0]}")

st.subheader("Prediction Probabilities")

proba_df = pd.DataFrame(
    proba[0],
    index=[f"Class {i}" for i in range(proba.shape[1])],
    columns=["Probability"]
)

st.bar_chart(proba_df)

st.subheader("Model Comparison")

results = pd.DataFrame({
    "Model": [
        "KNN",
        "Logistic Regression",
        "Decision Tree",
        "Random Forest"
    ],
    "CV Mean Accuracy": [
        0.971921,
        0.985961,
        0.916256,
        0.986207
    ],
    "CV Std": [
        0.025943,
        0.017199,
        0.051359,
        0.027586
    ]
})

st.dataframe(results)

best_model = results.loc[results["CV Mean Accuracy"].idxmax()]
st.markdown("### 🏆 Best Performing Model")
st.success(f"{best_model['Model']} (CV Mean: {best_model['CV Mean Accuracy']:.4f})")