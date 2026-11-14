"""Envía peticiones a la API para que los paneles de Grafana tengan datos.

Uso, con la API corriendo:  python src/generar_trafico.py
Opcional: otra dirección y otra cantidad, por ejemplo
    python src/generar_trafico.py http://localhost:8001 50

Solo usa la librería estándar de Python. CASO tiene las columnas de `Cliente` en
api/app.py (dataset student-mat, sin G3 ni las drop_columns), con valores reales
del dataset. La API responde la nota final predicha (G3, de 0 a 20).
"""
import json
import random
import sys
import time
import urllib.error
import urllib.request

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
CANTIDAD = int(sys.argv[2]) if len(sys.argv) > 2 else 20

# Primera fila de data/raw/student-mat.csv (sin school, reason, guardian, G1, G2, G3).
CASO = {
    "sex": "F", "age": 18, "address": "U", "famsize": "GT3", "Pstatus": "A",
    "Medu": 4, "Fedu": 4, "Mjob": "at_home", "Fjob": "teacher",
    "traveltime": 2, "studytime": 2, "failures": 0,
    "schoolsup": "yes", "famsup": "no", "paid": "no", "activities": "no",
    "nursery": "yes", "higher": "yes", "internet": "no", "romantic": "no",
    "famrel": 4, "freetime": 3, "goout": 4, "Dalc": 1, "Walc": 1, "health": 3,
    "absences": 6,
}

# Rangos permitidos por la API (los mismos Field(ge=..., le=...) de api/app.py).
RANGOS = {
    "age": (15, 22), "Medu": (0, 4), "Fedu": (0, 4),
    "traveltime": (1, 4), "studytime": (1, 4), "failures": (0, 3),
    "famrel": (1, 5), "freetime": (1, 5), "goout": (1, 5),
    "Dalc": (1, 5), "Walc": (1, 5), "health": (1, 5),
    "absences": (0, 75),
}

# Valores posibles de cada columna de texto, tal como aparecen en el dataset.
TRABAJOS = ["teacher", "health", "services", "at_home", "other"]
OPCIONES = {
    "sex": ["F", "M"], "address": ["U", "R"], "famsize": ["LE3", "GT3"],
    "Pstatus": ["T", "A"], "Mjob": TRABAJOS, "Fjob": TRABAJOS,
    **{c: ["no", "yes"] for c in ["schoolsup", "famsup", "paid", "activities",
                                   "nursery", "higher", "internet", "romantic"]},
}


def variar(caso):
    """Copia el caso y cambia cada columna dentro de lo que la API acepta,
    para que las predicciones cambien de una petición a otra."""
    nuevo = dict(caso)
    for columna, valor in caso.items():
        if columna in RANGOS:
            minimo, maximo = RANGOS[columna]
            if columna == "absences":
                # Las ausencias se concentran en valores bajos, como en el dataset.
                nuevo[columna] = min(maximo, int(random.expovariate(1 / 6)))
            else:
                nuevo[columna] = min(maximo, max(minimo, valor + random.randint(-1, 1)))
        elif columna in OPCIONES and random.random() < 0.3:
            nuevo[columna] = random.choice(OPCIONES[columna])
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
    if codigo == 200:
        print(f"{i:3d}  {codigo}  nota G3 predicha: {respuesta['prediccion']}")
    else:
        print(f"{i:3d}  {codigo}  {respuesta}")
    time.sleep(0.5)

# Un dato inválido: texto donde va un número. La API debe responder 422.
codigo, _ = enviar({**CASO, "studytime": "muchos"})
print(f"dato inválido en studytime: {codigo}")

# Falta un campo obligatorio (Mjob no tiene valor por defecto). También debe dar 422.
sin_mjob = {c: v for c, v in CASO.items() if c != "Mjob"}
codigo, _ = enviar(sin_mjob)
print(f"falta Mjob: {codigo}")
