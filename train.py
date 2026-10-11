
import argparse
import json
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report
)


# 1. Definir argumentos
def obtener_argumentos():
    parser = argparse.ArgumentParser(
        description="Entrenamiento Breast Cancer con MLflow"
    )

    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Parámetro de regularización"
    )

    parser.add_argument(
        "--max_iter",
        type=int,
        default=1000,
        help="Número máximo de iteraciones"
    )

    parser.add_argument(
        "--tol",
        type=float,
        default=0.0001,
        help="Tolerancia de convergencia"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla aleatoria"
    )

    return parser.parse_args()


# 2. Cargar y limpiar datos
def cargar_datos(seed):
    datos = load_breast_cancer(as_frame=True)

    X = datos.data.copy()

    # 1 = maligno, 0 = benigno
    y = (datos.target == 0).astype(int).rename("malignant")

    df = pd.concat([X, y], axis=1)
    df = df.drop_duplicates()

    X = df.drop(columns="malignant")
    y = df["malignant"]

    return train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=seed,
        stratify=y
    )


# 3. Entrenar modelo
def entrenar_modelo(X_train, y_train, args):
    modelo = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
        ("classifier", LogisticRegression(
            C=args.C,
            max_iter=args.max_iter,
            tol=args.tol,
            random_state=args.seed
        ))
    ])

    modelo.fit(X_train, y_train)

    return modelo


# 4. Evaluar modelo
def evaluar_modelo(modelo, X_test, y_test):
    predicciones = modelo.predict(X_test)
    probabilidades = modelo.predict_proba(X_test)[:, 1]

    metricas = {
        "accuracy": accuracy_score(y_test, predicciones),
        "precision_malignant": precision_score(y_test, predicciones),
        "recall_malignant": recall_score(y_test, predicciones),
        "f1_malignant": f1_score(y_test, predicciones),
        "roc_auc": roc_auc_score(y_test, probabilidades)
    }

    return metricas, predicciones, probabilidades


# 5. Guardar artefactos
def guardar_artefactos(y_test, predicciones, probabilidades, metricas):
    carpeta = Path("resultados")
    carpeta.mkdir(exist_ok=True)

    # Métricas en JSON
    archivo_metricas = carpeta / "metricas.json"

    with open(archivo_metricas, "w", encoding="utf-8") as f:
        json.dump(metricas, f, indent=4)

    # Predicciones en CSV
    archivo_predicciones = carpeta / "predicciones.csv"

    pd.DataFrame({
        "real": y_test.to_numpy(),
        "prediccion": predicciones,
        "probabilidad_maligno": probabilidades
    }).to_csv(archivo_predicciones, index=False)

    # Matriz de confusión
    matriz = confusion_matrix(y_test, predicciones)

    fig, ax = plt.subplots(figsize=(6, 5))

    ConfusionMatrixDisplay(
        confusion_matrix=matriz,
        display_labels=["Benigno", "Maligno"]
    ).plot(ax=ax, cmap="Blues", colorbar=False)

    ax.set_title("Matriz de confusión")

    archivo_matriz = carpeta / "matriz_confusion.png"
    fig.savefig(archivo_matriz, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Curva ROC
    fig, ax = plt.subplots(figsize=(6, 5))

    RocCurveDisplay.from_predictions(
        y_test,
        probabilidades,
        ax=ax
    )

    ax.set_title("Curva ROC")

    archivo_roc = carpeta / "curva_roc.png"
    fig.savefig(archivo_roc, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Reporte de clasificación
    reporte = classification_report(
        y_test,
        predicciones,
        target_names=["Benigno", "Maligno"],
        output_dict=True
    )

    archivo_reporte = carpeta / "classification_report.json"

    with open(archivo_reporte, "w", encoding="utf-8") as f:
        json.dump(reporte, f, indent=4)

    return carpeta


# 6. Ejecutar experimento
def main():
    args = obtener_argumentos()

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Breast_Cancer_Experimentacion_MLflow")

    X_train, X_test, y_train, y_test = cargar_datos(args.seed)

    with mlflow.start_run():

        # Registrar parámetros
        mlflow.log_params({
            "model": "LogisticRegression",
            "C": args.C,
            "max_iter": args.max_iter,
            "tol": args.tol,
            "seed": args.seed,
            "scaler": "RobustScaler",
            "imputer": "median",
            "test_size": 0.20
        })

        # Entrenar
        modelo = entrenar_modelo(
            X_train,
            y_train,
            args
        )

        # Evaluar
        metricas, predicciones, probabilidades = evaluar_modelo(
            modelo,
            X_test,
            y_test
        )

        # Registrar métricas
        mlflow.log_metrics(metricas)

        # Guardar y registrar artefactos
        carpeta = guardar_artefactos(
            y_test,
            predicciones,
            probabilidades,
            metricas
        )

        for archivo in carpeta.iterdir():
            if archivo.is_file():
                mlflow.log_artifact(
                    str(archivo),
                    artifact_path="resultados"
                )

        # Registrar modelo completo
        mlflow.sklearn.log_model(
            sk_model=modelo,
            name="logistic_regression",
            input_example=X_train.head(3),
            skops_trusted_types=["numpy.dtype"]
        )

        print("\nEntrenamiento finalizado")

        for nombre, valor in metricas.items():
            print(f"{nombre}: {valor:.4f}")


if __name__ == "__main__":
    main()
