"""Envía peticiones a la API para que los paneles de Grafana tengan datos.

Uso, con la API corriendo:  python src/generar_trafico.py
Opcional: otra dirección y otra cantidad, por ejemplo
    python src/generar_trafico.py http://localhost:8001 50

Solo usa la librería estándar de Python. CASO tiene las columnas del ejemplo de
churn: cámbienlas por las de `Cliente` en api/app.py, con valores reales del dataset.
"""
import json
import random
import sys
import time
import urllib.error
import urllib.request

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
CANTIDAD = int(sys.argv[2]) if len(sys.argv) > 2 else 20

CASO = {
    "sex": "F",            # "F" | "M"
    "age": 17,             # 15–22
    "address": "U",        # "U" urbano | "R" rural
    "famsize": "GT3",      # "LE3" | "GT3"
    "Pstatus": "T",        # "T" juntos | "A" separados
    "Medu": 3,             # 0–4
    "Fedu": 2,             # 0–4
    "Mjob": "services",    # obligatorio: teacher | health | services | at_home | other
    "Fjob": "other",       # obligatorio: teacher | health | services | at_home | other
    "traveltime": 1,       # 1–4
    "studytime": 2,        # 1–4
    "failures": 0,         # 0–3
    "schoolsup": "no",
    "famsup": "yes",
    "paid": "no",
    "activities": "yes",
    "nursery": "yes",
    "higher": "yes",
    "internet": "yes",
    "romantic": "no",
    "famrel": 4,           # 1–5
    "freetime": 3,         # 1–5
    "goout": 3,            # 1–5
    "Dalc": 1,             # 1–5
    "Walc": 2,             # 1–5
    "health": 4,           # 1–5
    "absences": 4,         # >= 0
}

def variar(caso):
    """Copia el caso y mueve un poco cada número para que las respuestas cambien."""
    nuevo = dict(caso)
    for columna, valor in caso.items():
        if isinstance(valor, int):
            nuevo[columna] = max(0, valor + random.randint(-10, 30))
        elif isinstance(valor, float):
            nuevo[columna] = round(max(0.0, valor * random.uniform(0.5, 1.5)), 2)
    return nuevo


def enviar(datos):
    peticion = urllib.request.Request(
        f"{URL}/predict", data=json.dumps(datos).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(peticion, timeout=10) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, e.reason


for i in range(1, CANTIDAD + 1):
    datos = variar(CASO)
    codigo, respuesta = enviar(datos)
    print(f"{i:3d}  {codigo}  {respuesta}")
    time.sleep(0.5)

# Un dato inválido: texto donde va un número. La API debe responder 422.
columna = next(c for c, v in CASO.items() if isinstance(v, (int, float)))
codigo, _ = enviar({**CASO, columna: "muchos"})
print(f"dato inválido en {columna}: {codigo}")
