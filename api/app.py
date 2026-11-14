"""API de inferencia: predice la nota final G3 de un estudiante."""

import csv
import os
from pathlib import Path
from threading import Lock
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import REGISTRY, Counter


modelo = joblib.load(Path(os.getenv("MODEL_DIR", "models")) / "model.pkl")
app = FastAPI(title="API de predicción de nota final G3 (student-mat)")

SiNo = Literal["no", "yes"]

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
    datos = cliente.model_dump()
    X = pd.get_dummies(pd.DataFrame([datos]), dtype=int)
    X = X.reindex(columns=modelo.feature_names_in_, fill_value=0)

    resultado = round(float(modelo.predict(X)[0]), 2)
    guardar_entrada(datos)
    PREDICCIONES.labels(clase=str(resultado)).inc()

    return {"prediccion": resultado}
