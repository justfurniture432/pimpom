"""API de inferencia: carga el modelo y responde si un cliente de Telco abandonará el servicio (Churn)."""
import os
from pathlib import Path
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

class Cliente(BaseModel):
    """Columnas del dataset (sin customerID ni Churn). Todo tiene un valor por defecto."""

    """Columnas del dataset sin el target (G3) ni las drop_columns
    (school, reason, guardian, G1, G2). Todo tiene un valor por defecto."""
 
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
 

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(cliente: Cliente):
    # Mismo one-hot que en src/prepare.py; reindex descarta las categorías de referencia
    # (drop_first) y rellena con 0 las columnas que el cliente no activa.
    X = pd.get_dummies(pd.DataFrame([cliente.model_dump()]), dtype=int)
    X = X.reindex(columns=modelo.feature_names_in_, fill_value=0)
    return {"prediccion": round(float(modelo.predict(X)[0]), 2)}




    
