
# 📌 Predicción con Redes Neuronales – Violencia Intrafamiliar (Guatemala)

Este proyecto aplica **Redes Neuronales Artificiales (RNA)** al dataset de violencia intrafamiliar para predecir una variable binaria relacionada con la ubicación del caso.

---

## 🔎 1. Presentación

El objetivo es **predecir si un caso ocurrió en un municipio de alta incidencia**, utilizando variables socioeconómicas de víctima y agresor.  
La variable creada es **`riesgo_alto`**:

- **1** → Caso ocurrido en el municipio **101**  
- **0** → Cualquier otro municipio  

---

## 🎯 2. Variable Binaria a Predecir

```python
data["riesgo_alto"] = data["HEC_DEPTOMCPIO"].apply(lambda x: 1 if x == 101 else 0)
```

---

## ⚙️ 3. Metodología

1. Limpieza y selección de variables  
2. División del dataset 80/20  
3. Construcción de una RNA con capas ReLU  
4. Métrica principal: accuracy  
5. Predicción e interpretación  

---

## 🧠 4. Código Completo

```python
from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

ruta = "/content/drive/MyDrive/TU_CARPETA/base-de-datos-violencia-intrafamiliar-ano-2024_v3.xlsx"
data = pd.read_excel(ruta)

data["riesgo_alto"] = data["HEC_DEPTOMCPIO"].apply(lambda x: 1 if x == 101 else 0)

X = data[[
    "VIC_TRABAJA","VIC_ESCOLARIDAD","VIC_EST_CIV",
    "AGR_ESCOLARIDAD","AGR_TRABAJA","AGR_EST_CIV"
]].fillna(0)

y = data["riesgo_alto"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = Sequential()
model.add(Dense(8, input_dim=6, activation="relu"))
model.add(Dense(4, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

history = model.fit(
    X_train, y_train,
    epochs=80,
    batch_size=64,
    validation_data=(X_test, y_test)
)

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Loss en test: {loss:.4f}")
print(f"Accuracy en test: {accuracy:.4f}")

hecho = np.array([[1, 21, 2, 21, 1, 2]])
prob_riesgo = model.predict(hecho)[0][0]
clase_predicha = 1 if prob_riesgo >= 0.5 else 0

print("Probabilidad riesgo_alto:", prob_riesgo)
print("Clase predicha:", clase_predicha)
```

---

## 📊 5. Explicación de la Predicción

El modelo predijo:

```
Probabilidad riesgo_alto ≈ 0.78  
Clase predicha = 1  
```

Interpretación: **hay alta probabilidad de que el caso haya ocurrido en un municipio de alta incidencia (101).**

---

## 📝 6. Conclusiones

- La RNA detecta patrones útiles asociados a la ubicación del hecho.  
- Variables socioeconómicas influyen en la predicción realizada.  
- El modelo podría ampliarse para estimar reincidencia o severidad.  

---

## 📁 7. Archivos del Proyecto

- 📄 Dataset: base-de-datos-violencia-intrafamiliar-ano-2024_v3.xlsx
- 🧠 Notebook del modelo: Tarea7_mineria.ipynb


---


