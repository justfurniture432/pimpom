"""Compara los datos de entrenamiento con los recibidos por la API.

Uso:
    python src/drift.py
    python src/drift.py logs/actual.csv

Genera:
    reports/drift.html
"""

import sys
from pathlib import Path

import pandas as pd
import yaml
from evidently import Report
from evidently.presets import DataDriftPreset


ACTUAL = Path(sys.argv[1] if len(sys.argv) > 1 else "logs/actual.csv")

# Ajusta esta ruta al archivo usado para entrenar tu modelo.
REFERENCIA = Path("data/raw/student-mat.csv")

if not REFERENCIA.is_file():
    raise SystemExit(f"No se encontró el dataset de referencia: {REFERENCIA}")

if not ACTUAL.is_file():
    raise SystemExit(
        f"No se encontró {ACTUAL}. Ejecuta primero simular_drift.py."
    )

with open("params.yaml", encoding="utf-8") as archivo:
    params = yaml.safe_load(archivo)["prepare"]

# El dataset original de estudiantes usa punto y coma como separador.
referencia = pd.read_csv(REFERENCIA, sep=None, engine="python")
referencia = referencia.drop(
    columns=[params["target"], *params["drop_columns"]],
    errors="ignore",
)

actual = pd.read_csv(ACTUAL)

columnas = [columna for columna in referencia.columns if columna in actual.columns]

if not columnas:
    raise SystemExit(
        "No hay columnas comunes entre el dataset de referencia y los datos actuales."
    )

if referencia.empty or actual.empty:
    raise SystemExit("Los dos archivos deben contener filas de datos.")

referencia = referencia[columnas]
actual = actual[columnas]

informe = Report([DataDriftPreset()]).run(
    reference_data=referencia,
    current_data=actual,
)

Path("reports").mkdir(parents=True, exist_ok=True)
informe.save_html("reports/drift.html")

print(f"Columnas comparadas: {len(columnas)}")
print("Informe generado: reports/drift.html")
