import importlib
import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(_file_).resolve().parents[1]))


@pytest.fixture()
def client(tmp_path, monkeypatch):
    # Modelo de prueba: predice G3 a partir del tiempo de estudio.
    X = pd.DataFrame({"studytime": [1, 4]})
    y = [8.0, 17.0]

    modelo = LinearRegression().fit(X, y)
    joblib.dump(modelo, tmp_path / "model.pkl")

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
    r = client.post("/predict", json={"studytime": 2})
    assert r.status_code == 200
    assert r.json()["prediccion"] == pytest.approx(11.0)


def test_dato_invalido(client):
    r = client.post("/predict", json={"studytime": "muchos"})
    assert r.status_code == 422