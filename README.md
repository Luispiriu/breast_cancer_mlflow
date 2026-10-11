# breast_cancer_mlflow

# Experimentación de Machine Learning con MLflow

## Descripción del proyecto

Este proyecto consiste en la automatización del entrenamiento de un modelo de clasificación de cáncer de mama utilizando el dataset Breast Cancer Wisconsin y un modelo de regresión logística.

El entrenamiento se realiza mediante un script de Python parametrizado con `argparse`, mientras que MLflow permite registrar y comparar los parámetros, las métricas, los modelos y los artefactos generados en cada ejecución.

## Estructura del proyecto

```text
Proyecto_MLflow/
├── train.py
├── requirements.txt
├── README.md
└── Tarea.pdf
```

## Instalación

**1. Crear y activar un entorno virtual**

```bash
conda create -n mlflow python=3.11 -y
conda activate mlflow
```

**2. Instalar las dependencias**

```bash
python -m pip install -r requirements.txt
```

**3. Iniciar el servidor de MLflow**

En una terminal:

```bash
mlflow server --host 127.0.0.1 --port 5000
```

**4. Ejecutar el entrenamiento**

En otra terminal, con el entorno activado:

```bash
python train.py --C 1.0 --max_iter 1000 --tol 0.0001 --seed 42
```

Para realizar las cinco configuraciones utilizadas en la actividad:

```bash
python train.py --C 1.0 --max_iter 1000 --tol 0.0001 --seed 42
python train.py --C 0.1 --max_iter 1000 --tol 0.0001 --seed 42
python train.py --C 10.0 --max_iter 1000 --tol 0.0001 --seed 42
python train.py --C 1.0 --max_iter 200 --tol 0.0001 --seed 42
python train.py --C 1.0 --max_iter 1000 --tol 0.001 --seed 42
```

## Seguimiento de experimentos

La interfaz de MLflow se encuentra disponible en:

http://127.0.0.1:5000

Experimento: `Breast_Cancer_Experimentacion_MLflow`

En cada ejecución se registran:

- **Parámetros:** `C`, `max_iter`, `tol` y `seed`.
- **Métricas:** accuracy, precision, recall, F1-score y ROC-AUC.
- **Artefactos:** matriz de confusión, curva ROC, predicciones y reportes de evaluación.
- **Modelo:** pipeline entrenado de regresión logística.

## Resultados

Se realizaron cinco corridas para evaluar diferentes configuraciones del modelo.

La configuración base obtuvo un accuracy de 0.9825, un F1-score de 0.9756 y un ROC-AUC aproximado de 0.9967.

Al comparar las ejecuciones, se observó que el hiperparámetro `C` tuvo mayor influencia sobre accuracy y F1-score, mientras que los cambios en `max_iter` y `tol` no produjeron diferencias importantes en estas métricas.

## Reproducibilidad

Se utilizó una semilla de 42 para mantener la misma división de los datos y facilitar la reproducción de los experimentos. Las versiones de las dependencias se encuentran especificadas en `requirements.txt`.

El entrenamiento puede repetirse utilizando los comandos anteriores, una vez instalado el entorno y puesto en funcionamiento el servidor de MLflow.

## Documentación

El archivo `Tarea.pdf` contiene la descripción del experimento, las evidencias de MLflow y el análisis de los resultados.
