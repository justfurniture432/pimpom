"""API de inferencia: carga el modelo y responde si un cliente de Telco abandonará el servicio (Churn)."""
import csv
import os
from pathlib import Path
from threading import Lock
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from prometheus_client import REGISTRY, Counter
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import REGISTRY, Counter


LOGS = Path(os.getenv("LOGS_DIR", "logs"))
modelo = joblib.load(Path(os.getenv("MODEL_DIR", "models")) / "model.pkl")
#---
app = FastAPI(title="API de predicción de nota final G3 (student-mat)")
Instrumentator().instrument(app).expose(app)

try:
    PREDICCIONES = Counter("predicciones_total", "Predicciones por clase", ["clase"])
except ValueError:  # ya existe: pasa cuando las pruebas vuelven a cargar la API
    PREDICCIONES = REGISTRY._names_to_collectors["predicciones_total"]


def guardar_entrada(datos):
    """Agrega una fila a logs/entradas.csv con lo que recibió la API."""
    LOGS.mkdir(exist_ok=True)
    archivo = LOGS / "entradas.csv"
    nuevo = not archivo.exists()
    with open(archivo, "a", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=list(datos))
        if nuevo:
            escritor.writeheader()
        escritor.writerow(datos)


SiNo = Literal["no", "yes"]
SiNoSinInternet = Literal["no", "yes", "no internet service"]

Instrumentator().instrument(app).expose(app)

try:
    PREDICCIONES = Counter(
        "predicciones_total",
        "Predicciones por clase",
        ["clase"]
    )
except ValueError:
    PREDICCIONES = REGISTRY._names_to_collectors["predicciones_total"]
ENTRADAS_CSV = Path("logs") / "entradas.csv"
ENTRADAS_LOCK = Lock()


class Cliente(BaseModel):
    """Variables de entrada sin G3, school, reason, guardian, G1 ni G2."""

    sex: Literal["F", "M"] = "F"
    age: int = Field(17, ge=15, le=22, description="Edad del estudiante")
    address: Literal["U", "R"] = "U"
    famsize: Literal["LE3", "GT3"] = "GT3"
    Pstatus: Literal["T", "A"] = "T"
    Medu: int = Field(3, ge=0, le=4, description="Educación de la madre (0-4)")
    Fedu: int = Field(2, ge=0, le=4, description="Educación del padre (0-4)")
    Mjob: str
    Fjob: str
    traveltime: int = Field(1, ge=1, le=4)
    studytime: int = Field(2, ge=1, le=4)
    failures: int = Field(0, ge=0, le=3)
    schoolsup: SiNo = "no"
    famsup: SiNo = "yes"
    paid: SiNo = "no"
    activities: SiNo = "yes"
    nursery: SiNo = "yes"
    higher: SiNo = "yes"
    internet: SiNo = "yes"
    romantic: SiNo = "no"
    famrel: int = Field(4, ge=1, le=5)
    freetime: int = Field(3, ge=1, le=5)
    goout: int = Field(3, ge=1, le=5)
    Dalc: int = Field(1, ge=1, le=5)
    Walc: int = Field(2, ge=1, le=5)
    health: int = Field(4, ge=1, le=5)
    absences: int = Field(4, ge=0, description="Número de ausencias")


def guardar_entrada(datos: dict):
    """Agrega una entrada al CSV sin borrar las filas anteriores."""
    with ENTRADAS_LOCK:
        ENTRADAS_CSV.parent.mkdir(parents=True, exist_ok=True)

        escribir_cabecera = (
            not ENTRADAS_CSV.exists()
            or ENTRADAS_CSV.stat().st_size == 0
        )

        with ENTRADAS_CSV.open("a", newline="", encoding="utf-8") as archivo:
            escritor = csv.DictWriter(
                archivo,
                fieldnames=list(datos.keys()),
            )
            if escribir_cabecera:
                escritor.writeheader()

            escritor.writerow(datos)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(cliente: Cliente):
    guardar_entrada(cliente.model_dump())
    # Mismo one-hot que en src/prepare.py; reindex descarta las categorías de referencia
    # (drop_first) y rellena con 0 las columnas que el cliente no activa.
    X = pd.get_dummies(pd.DataFrame([cliente.model_dump()]), dtype=int)
    X = X.reindex(columns=modelo.feature_names_in_, fill_value=0)
    #return {"prediccion": round(float(modelo.predict(X)[0]), 2)}   
    resultado = round(float(modelo.predict(X)[0]), 2)
    PREDICCIONES.labels(clase=resultado).inc()

    return {"prediccion": resultado} 

