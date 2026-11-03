"""Etapa train: entrena el modelo y lo guarda en models/model.pkl."""
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier

params = yaml.safe_load(open("params.yaml", encoding="utf-8"))
target = params["prepare"]["target"]
p = params["train"]

train = pd.read_csv("data/prepared/train.csv")
X, y = train.drop(columns=[target]), train[target]

model = RandomForestClassifier(
    n_estimators=p["n_estimators"],
    max_depth=p["max_depth"],
    random_state=params["prepare"]["random_state"],
)
model.fit(X, y)

Path("models").mkdir(exist_ok=True)
joblib.dump(model, "models/model.pkl")
print("Modelo guardado en models/model.pkl")
