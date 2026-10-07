import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

with np.load(BASE_DIR / "data" / "data.npz") as data:
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]


models = {
    "KNN": make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=5),
    ),
    "SVM": make_pipeline(
        StandardScaler(),
        SVC(),
    ),
}


def evaluate(model, Xtr, ytr, Xte, yte):
    fit_times, predict_times = [], []

    for _ in range(10):
        start = time.perf_counter()
        model.fit(Xtr, ytr)
        fit_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        pred = model.predict(Xte)
        predict_times.append(time.perf_counter() - start)

    tracemalloc.start()
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    memory_kb = tracemalloc.get_traced_memory()[1] / 1024
    tracemalloc.stop()

    return {
        "features": Xtr.shape[1],
        "accuracy": accuracy_score(yte, pred),
        "precision": precision_score(yte, pred, average="macro"),
        "recall": recall_score(yte, pred, average="macro"),
        "f1": f1_score(yte, pred, average="macro"),
        "train_time": np.mean(fit_times),
        "predict_time": np.mean(predict_times),
        "memory_KB": memory_kb,
    }


results = {
    name: evaluate(model, X_train, y_train, X_test, y_test)
    for name, model in models.items()
}

table = pd.DataFrame.from_dict(results, orient="index")
table.to_csv(RESULTS_DIR / "baseline.csv")

print(table.round(4))