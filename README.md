# 🍷 Proyecto de Clasificación de Vinos

---

## Descripción

Este proyecto desarrolla un enfoque de Machine Learning supervisado para clasificar vinos a partir de su composición química.

A través de análisis estadístico, exploración visual y reducción de dimensionalidad, se evalúa si las distintas clases de vino son estructuralmente separables en el espacio de características.

El dataset contiene resultados de análisis químicos de vinos cultivados en la región de Piamonte (Italia), pertenecientes a tres cultivares distintos.

---

## Objetivos

- Analizar la estructura química de las distintas clases de vino.
- Identificar las variables más discriminantes mediante ANOVA (F-test).
- Explorar distribuciones y relaciones entre variables.
- Evaluar la separabilidad estructural utilizando PCA.
- Desarrollar y evaluar un modelo de clasificación multiclase (en progreso).

---

## Metodología

1. Carga y preprocesamiento de datos  
2. Análisis exploratorio (EDA)  
3. Identificación de variables discriminantes (ANOVA)  
4. Visualización de distribuciones (boxplots y scatter plots)  
5. Reducción de dimensionalidad con PCA  
6. Desarrollo y evaluación del modelo (siguiente fase)

---

## Resultados preliminares

- **Flavanoids** y **Proline** son las variables con mayor capacidad discriminante.
- **Alcohol** contribuye significativamente a la diferenciación entre clases.
- El análisis ANOVA confirma diferencias estadísticamente significativas.
- La proyección PCA en dos dimensiones explica aproximadamente el 55% de la varianza total y muestra una clara separabilidad geométrica entre clases.

Estos resultados sugieren que la composición química por sí sola permite distinguir estructuralmente los perfiles de vino.

---

## Estructura del proyecto

Wine-ML/
│
├── data/  
├── notebooks/  
│   ├── 01_Wine_ML.ipynb  
│   └── 02_Wine_Profile.ipynb  
├── src/  
│   └── functions.py  
├── README.md  

---

## Tecnologías utilizadas

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  
- Seaborn  

---

## Próximos pasos

- Implementar modelos supervisados de clasificación.
- Evaluar rendimiento mediante train/test split.
- Comparar distintos algoritmos.
- Analizar importancia de variables.
- Mejorar interpretabilidad del modelo.

---

## Autor

Lautaro DA  
Machine Learning & Análisis de Datos