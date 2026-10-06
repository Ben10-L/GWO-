import os, time, tracemalloc
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

os.makedirs("results", exist_ok=True)

df = pd.read_csv("movement_libras.data", header=None)
X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)
np.savez("data/data.npz", X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
print(X_train.shape, X_test.shape)

def evaluate(model, Xtr, ytr, Xte, yte):
    ft, pt = [], []
    for _ in range(10):
        t = time.perf_counter(); model.fit(Xtr, ytr); ft.append(time.perf_counter() - t)
        t = time.perf_counter(); pred = model.predict(Xte); pt.append(time.perf_counter() - t)
    tracemalloc.start()
    model.fit(Xtr, ytr); model.predict(Xte)
    mem = tracemalloc.get_traced_memory()[1] / 1024
    tracemalloc.stop()
    return {"features": Xtr.shape[1],
            "accuracy": accuracy_score(yte, pred),
            "precision": precision_score(yte, pred, average="macro", zero_division=0),
            "recall": recall_score(yte, pred, average="macro", zero_division=0),
            "f1": f1_score(yte, pred, average="macro", zero_division=0),
            "train_time": np.mean(ft), "predict_time": np.mean(pt), "memory_KB": mem}, pred

models = {"KNN": KNeighborsClassifier(n_neighbors=5), "SVM": SVC()}
rows = {}
for name, model in models.items():
    rows[name], _ = evaluate(model, X_train, y_train, X_test, y_test)

baseline = pd.DataFrame(rows).T
baseline.to_csv("results/baseline.csv")
baseline.round(4)
