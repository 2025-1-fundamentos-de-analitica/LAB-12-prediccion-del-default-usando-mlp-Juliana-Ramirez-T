# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.



import pandas as pd
import numpy as np
import os
import json
import gzip
import pickle

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
)


def pregunta_01():
    # Paso 1.
    #Cargar datoss
    train = pd.read_csv("files/input/train_data.csv.zip", compression="zip", index_col=False)
    test = pd.read_csv("files/input/test_data.csv.zip", compression="zip", index_col=False)
    # - Renombre la columna "default payment next month" a "default".
    train.rename(columns={"default payment next month": "default"}, inplace=True)
    test.rename(columns={"default payment next month": "default"}, inplace=True)
    # - Remueva la columna "ID".
    train.drop(columns=["ID"], inplace=True)
    test.drop(columns=["ID"], inplace=True)
    # - Para la columna EDUCATION, valores > 4 indican niveles superiores de educación, agrupe estos valores en la categoría "others".
    train = train[(train["MARRIAGE"] != 0) & (train["EDUCATION"] != 0)]
    test = test[(test["MARRIAGE"] != 0) & (test["EDUCATION"] != 0)]
    train["EDUCATION"] = train["EDUCATION"].apply(lambda x: 4 if x > 4 else x)
    test["EDUCATION"] = test["EDUCATION"].apply(lambda x: 4 if x > 4 else x)
    # Paso 2.
    # Divida los datasets en x_train, y_train, x_test, y_test.
    x_train, y_train = train.drop(columns="default"), train["default"]
    x_test, y_test = test.drop(columns="default"), test["default"]
    # Paso 3.
    # Cree un pipeline para el modelo de clasificación. Este pipeline debe
    # contener las siguientes capas:
    # - Transforma las variables categoricas usando el método
    #   one-hot-encoding.
    # - Descompone la matriz de entrada usando componentes principales.
    #   El pca usa todas las componentes.
    # - Escala la matriz de entrada al intervalo [0, 1].
    # - Selecciona las K columnas mas relevantes de la matrix de entrada.
    # - Ajusta una red neuronal tipo MLP.
    cat_vars = ["SEX", "EDUCATION", "MARRIAGE"]
    num_vars = [col for col in x_train.columns if col not in cat_vars]
    transformador = ColumnTransformer(transformers=[("cat", OneHotEncoder(), cat_vars),("scaler", StandardScaler(), num_vars),])
    modelo_pipe = Pipeline([
        ("transf", transformador),
        ("pca", PCA()),
        ("selec_caract", SelectKBest(score_func=f_classif)),
        ("clasificador", MLPClassifier(max_iter=300, random_state=12345)),
    ])
    # Paso 4.
    # Optimice los hiperparametros del pipeline usando validación cruzada.
    # Use 10 splits para la validación cruzada. Use la función de precision balanceada para medir la precisión del modelo.
    paramet = {
        "pca__n_components": [20, x_train.shape[1] - 2],
        "feature_selection__k": [12],
        "classifier__hidden_layer_sizes": [(100,), (50, 50)],
        "classifier__alpha": [0.0001, 0.001],
    }
    estima = GridSearchCV(
        modelo_pipe,
        paramet,
        cv=10,
        scoring="balanced_accuracy",
        n_jobs=-1,
        refit=True,
    )
    estima.fit(x_train, y_train)
    # Paso 5.
    # Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
    # Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
    os.makedirs("files/models", exist_ok=True)
    with gzip.open("files/models/model.pkl.gz", "wb") as f:
        pickle.dump(estima, f)
    # Paso 6.
    # Calcule las metricas de precision, precision balanceada, recall,
    # y f1-score para los conjuntos de entrenamiento y prueba.
    # Guardelas en el archivo files/output/metrics.json. Cada fila
    # del archivo es un diccionario con las metricas de un modelo.
    # Este diccionario tiene un campo para indicar si es el conjunto de entrenamiento o prueba.
    def cmetricas(tipo, y_true, y_pred):
        return {
            "type": "metrics",
            "dataset": tipo,
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
        }
    # Paso 7.
    # Calcule las matrices de confusion para los conjuntos de entrenamiento y
    # prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
    # del archivo es un diccionario con las metricas de un modelo.
    # de entrenamiento o prueba.
    def matrizconfu(tipo, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        return {
            "type": "cm_matrix",
            "dataset": tipo,
            "true_0": {"predicted_0": int(cm[0, 0]), "predicted_1": int(cm[0, 1])},
            "true_1": {"predicted_0": int(cm[1, 0]), "predicted_1": int(cm[1, 1])},
        }

    y_train_pred = estima.predict(x_train)
    y_test_pred = estima.predict(x_test)

    final = [
        cmetricas("train", y_train, y_train_pred),
        cmetricas("test", y_test, y_test_pred),
        matrizconfu("train", y_train, y_train_pred),
        matrizconfu("test", y_test, y_test_pred),
    ]

    os.makedirs("files/output", exist_ok=True)
    with open("files/output/metrics.json", "w", encoding="utf-8") as file:
        for i in final:
            file.write(json.dumps(i) + "\n")


