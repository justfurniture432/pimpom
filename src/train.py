"""Etapa train: entrena el modelo, lo guarda y lo registra en MLflow."""
import time
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

params = yaml.safe_load(open("params.yaml", encoding="utf-8"))
target = params["prepare"]["target"]
p = params["train"]

train = pd.read_csv("data/prepared/train.csv")
X, y = train.drop(columns=[target]), train[target]

mlflow.set_experiment("random-forest-regresion")

with mlflow.start_run():
    mlflow.log_params({
        "n_estimators": p["n_estimators"],
        "max_depth": p["max_depth"],
        "random_state": params["prepare"]["random_state"],
        "target": target,
    })

    model = RandomForestRegressor(
        n_estimators=p["n_estimators"],
        max_depth=p["max_depth"],
        random_state=params["prepare"]["random_state"],
    )


    start = time.perf_counter()
    model.fit(X, y)
    train_time = time.perf_counter() - start

    pred_train = model.predict(X)

    mlflow.log_metrics({
        "train_time_s": train_time,
        "train_mae": mean_absolute_error(y, pred_train),
        "train_rmse": mean_squared_error(y, pred_train) ** 0.5,
        "train_r2": r2_score(y, pred_train),
    })

    mlflow.sklearn.log_model(
        model,
        name="model",
        skops_trusted_types=["sklearn.tree._tree.Tree"],
    )

    # DVC sigue usando este archivo como salida de la etapa
    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, "models/model.pkl")

print("Modelo guardado en models/model.pkl")