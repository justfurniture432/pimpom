"""Prueba el endpoint /predict de la API (puerto 8000) usando curl.

Ejecutar con la API corriendo:
    python probar_predict.py
"""
import json
import subprocess

URL = "http://localhost:8000/predict"

# Mjob y Fjob son obligatorios (no tienen valor por defecto).
# Valores válidos del dataset: "teacher", "health", "services", "at_home", "other".
# Los campos sí/no (schoolsup, famsup, paid, ...) se omiten a propósito:
# así la API usa sus valores por defecto en minúscula ("yes"/"no"), que son
# los que coinciden con las columnas con las que se entrenó el modelo.
estudiante = {
    "sex": "F",
    "age": 17,
    "address": "U",
    "famsize": "GT3",
    "Pstatus": "T",
    "Medu": 4,
    "Fedu": 3,
    "Mjob": "teacher",
    "Fjob": "services",
    "traveltime": 1,
    "studytime": 3,
    "failures": 0,
    "famrel": 4,
    "freetime": 3,
    "goout": 2,
    "Dalc": 1,
    "Walc": 1,
    "health": 5,
    "absences": 2,
}

cmd = [
    "curl", "-s", "-X", "POST", URL,
    "-H", "Content-Type: application/json",
    "-d", json.dumps(estudiante),
]

print("Comando:\n", " ".join(cmd[:-1]), f"'{cmd[-1]}'\n")

resultado = subprocess.run(cmd, capture_output=True, text=True)

if resultado.returncode != 0:
    print("curl falló (¿está la API corriendo en el puerto 8000?)")
    print(resultado.stderr)
else:
    try:
        print("Respuesta:\n", json.dumps(json.loads(resultado.stdout), indent=2, ensure_ascii=False))
    except json.JSONDecodeError:
        print("Respuesta (texto):\n", resultado.stdout)
