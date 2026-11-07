import importlib
import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Mjob y Fjob son obligatorios en la API (no tienen valor por defecto).
OBLIGATORIOS = {"Mjob": "teacher", "Fjob": "services"}


@pytest.fixture()
def client(tmp_path, monkeypatch):
    # Modelo de prueba: predice G3 (nota 0-20) a partir del tiempo de estudio.
    X = pd.DataFrame({"studytime": [1, 4]})
    y = [8.0, 17.0]
    joblib.dump(LinearRegression().fit(X, y), tmp_path / "model.pkl")

    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    import api.app as modulo
    importlib.reload(modulo)

    with TestClient(modulo.app) as cliente:
        yield cliente


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict(client):
    r = client.post("/predict", json={**OBLIGATORIOS, "studytime": 2})
    assert r.status_code == 200, r.text
    # Recta que pasa por (1, 8) y (4, 17): studytime=2 -> 11
    assert r.json()["prediccion"] == pytest.approx(11.0)


def test_dato_invalido(client):
    r = client.post("/predict", json={**OBLIGATORIOS, "studytime": "muchos"})
    assert r.status_code == 422


def test_falta_campo_obligatorio(client):
    r = client.post("/predict", json={"studytime": 2})
    assert r.status_code == 422
