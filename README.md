# 🍷 Wine Classification Simulator

Proyecto de Machine Learning enfocado en la clasificación de vinos a partir de su composición química, combinando análisis estructural exploratorio y validación rigurosa de modelos supervisados.


---

## 🎯 Objetivo

Clasificar tres tipos de vino del dataset UCI Wine y analizar las diferencias estructurales entre clases, priorizando no solo rendimiento predictivo sino también estabilidad e interpretabilidad del modelo.

---

## 🔬 Análisis Exploratorio

Se realizaron:

- Perfil promedio por clase

- ANOVA (F-score) para evaluar poder discriminante de cada variable

- PCA para visualizar separabilidad estructural en dos dimensiones

Las variables con mayor capacidad de separación fueron:

Flavanoides, Prolina, OD280/OD315, Alcohol e Intensidad de color.

La proyección PCA confirmó la existencia de estructura diferenciada entre clases.

---

## 🤖 Estrategia de Modelado

Metodología aplicada:

- División Train/Test (80% / 20%)

- 5-Fold Cross Validation sobre el conjunto de entrenamiento

- Evaluación de:

  . Accuracy media

  . Desviación estándar

  . Complejidad del modelo

  . Interpretabilidad

- Modelos evaluados:

  . KNN

  . Decision Tree

  . Random Forest

  . Logistic Regression

---

## 🏆 Selección del Modelo

Se seleccionó Logistic Regression por ofrecer:

- Alto rendimiento consistente

- Menor variabilidad entre folds

- Menor complejidad estructural

- Alta interpretabilidad de coeficientes

El criterio priorizó robustez y claridad sobre complejidad innecesaria.

---

## 🚀 Aplicación Interactiva

El modelo fue integrado en una aplicación desarrollada con Streamlit que permite:

- Ajustar dinámicamente la composición química

- Generar un “Random Realistic Wine”

- Visualizar probabilidades de clasificación

- Interpretar el perfil estructural resultante

La herramienta transforma el modelo en un entorno exploratorio y explicativo.

---

## 🛠 Tecnologías

Python · Pandas · Scikit-learn · Matplotlib · Streamlit

---

## ▶️ Ejecución
streamlit run app.py

---

## Autor
Lautaro Silvestri

Machine Learning & Data Science
